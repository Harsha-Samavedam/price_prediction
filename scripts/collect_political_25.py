import csv
import sys
import time
from datetime import timedelta
from pathlib import Path

from claim_reaction import sources
from claim_reaction.core import encoded_array, iso, parse_time
from claim_reaction.store import Store


ROOT = Path(__file__).resolve().parents[1] / 'data'
ROOT.mkdir(parents=True, exist_ok=True)
PRICES_ONLY = '--prices-only' in sys.argv[1:]

# One market per political event where possible, spanning multiple countries
# and policy, election, and geopolitical questions.
SELECTION = [
    ('will-there-be-no-change-in-fed-interest-rates-after-the-july-2026-meeting', 'Federal Reserve interest rates July 2026', 'monetary_policy'),
    ('will-the-fed-decrease-interest-rates-by-25-bps-after-the-september-2026-meeting-586', 'Federal Reserve interest rates September 2026', 'monetary_policy'),
    ('will-the-senate-vote-on-the-clarity-act-before-the-august-recess-20260729185025133', 'CLARITY Act Senate vote', 'us_legislation'),
    ('scotus-bars-counting-mail-ballots-after-election-day', 'Supreme Court mail ballots election day', 'us_courts'),
    ('us-x-iran-effective-ceasefire-by-september-4', '"Iran ceasefire"', 'diplomacy_conflict'),
    ('us-announces-end-of-iranian-blockade-by-september-30-2026-20260727171615364-722-649-646-561-213-644-414-831', '"Iran blockade"', 'diplomacy_conflict'),
    ('russia-x-ukraine-ceasefire-agreement-by-august-31-2026', '"Ukraine ceasefire"', 'diplomacy_conflict'),
    ('ukraine-election-called-by-august-31-2026', '"Ukraine election"', 'elections'),
    ('will-china-invade-taiwan-by-september-30-2026', '"Taiwan invasion"', 'diplomacy_conflict'),
    ('israel-closes-its-airspace-by-september-30', 'Israel airspace closure', 'diplomacy_conflict'),
    ('strait-of-hormuz-traffic-returns-to-normal-by-august-31-20260702154212320', '"Strait of Hormuz"', 'diplomacy_conflict'),
    ('will-the-iranian-regime-fall-by-september-30-20260629143335077', 'Iran regime government', 'leadership'),
    ('will-lula-win-the-most-votes-in-the-first-round-of-the-2026-brazil-presidential-election', '"Lula" "Brazil election"', 'elections'),
    ('will-united-russia-er-gain-the-most-seats-in-the-next-russian-parliamentary-election', '"United Russia" election', 'elections'),
    ('will-the-sweden-democrats-sd-win-the-most-seats-in-the-2026-swedish-parliamentary-election', '"Sweden Democrats" election', 'elections'),
    ('will-afd-win-the-most-seats-in-the-2026-berlin-state-elections', 'Berlin state election AfD', 'elections'),
    ('will-cdu-win-the-most-seats-in-the-2026-sachsen-anhalt-parliamentary-elections', '"Saxony Anhalt" election', 'elections'),
    ('will-james-fishback-be-the-republican-nominee-for-florida-governor', '"James Fishback"', 'elections'),
    ('will-francesca-hong-win-the-2026-wisconsin-governor-democratic-primary-election', '"Francesca Hong"', 'elections'),
    ('will-peggy-flanagan-be-the-democratic-nominee-for-senate-in-minnesota', '"Peggy Flanagan"', 'elections'),
    ('will-parti-qubcois-win-the-most-seats-in-the-2026-quebec-general-election', '"Parti Quebecois"', 'elections'),
    ('will-nigel-farage-win-the-clacton-by-election-20260707144027825', '"Nigel Farage" Clacton', 'elections'),
    ('will-a-new-cabinet-of-the-netherlands-be-sworn-in-by-june-30-2026-492', 'Netherlands new cabinet', 'leadership'),
    ('us-x-yemen-ceasefire-continues-through-october-15', 'United States Yemen ceasefire', 'diplomacy_conflict'),
    ('will-maria-corina-machado-enter-venezuela-by-july-31-20260626160132235', '"Maria Corina Machado"', 'leadership'),
]


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def yes_outcome_price(raw):
    try:
        labels = encoded_array(raw['outcomes'])
        values = encoded_array(raw['outcomePrices'])
        return float(values[[str(x).lower() for x in labels].index('yes')])
    except (ValueError, KeyError, IndexError, TypeError):
        return None


original_get_json = sources.get_json
last_gdelt_request = 0.0


def paced_get_json(url, params=None, headers=None):
    global last_gdelt_request
    if 'api.gdeltproject.org' not in url:
        return original_get_json(url, params, headers)
    pause = max(0, 12.0 - (time.monotonic() - last_gdelt_request))
    if pause:
        time.sleep(pause)
    last_gdelt_request = time.monotonic()
    return original_get_json(url, params, headers)


def gdelt_sample(query, start, end, slug):
    # A single bounded historical search per market. Record the returned count
    # and cap explicitly; this is an initial candidate set, not exhaustive news.
    response = paced_get_json('https://api.gdeltproject.org/api/v2/doc/doc', {
        'query': query,
        'mode': 'artlist',
        'format': 'json',
        'startdatetime': (start - timedelta(seconds=1)).strftime('%Y%m%d%H%M%S'),
        'enddatetime': end.strftime('%Y%m%d%H%M%S'),
        'maxrecords': 250,
    })
    items = []
    for article in response.get('articles', []):
        url = article.get('url')
        seen = sources.gdelt_time(article.get('seendate'))
        if url and seen and start <= seen < end:
            items.append(sources.item('gdelt-doc', url, slug, query, url,
                                      article.get('title') or '', None, seen, article))
    return items, len(response.get('articles', [])) == 250


polymarket = sources.Polymarket()
previous = {}
if (ROOT / 'coverage.csv').exists():
    with (ROOT / 'coverage.csv').open(newline='', encoding='utf-8') as handle:
        previous = {row['market_slug']: row for row in csv.DictReader(handle)}


def save_coverage(completed):
    current = {row['market_slug']: row for row in completed}
    ordered = [current.get(slug) or previous[slug]
               for slug, _, _ in SELECTION if slug in current or slug in previous]
    if ordered:
        write_csv(ROOT / 'coverage.csv', list(ordered[0]), ordered)


def needs_deadline_window(row):
    if not row['scheduled_end'] or not row['closed_time']:
        return False
    scheduled = parse_time(row['scheduled_end'])
    closed = parse_time(row['closed_time'])
    return closed - scheduled > timedelta(days=1) and row['window_end'] != iso(scheduled)


rows = []
with Store(str(ROOT / 'political_25.sqlite')) as store:
    for number, (slug, query, topic) in enumerate(SELECTION, 1):
        row = dict(previous.get(slug) or {'market_slug': slug, 'topic': topic, 'news_query': query,
               'market_api_url': polymarket.GAMMA + '/markets/slug/' + slug,
               'question': '', 'window_start': '', 'window_end': '',
               'scheduled_end': '', 'closed_time': '',
               'resolution_status': '', 'yes_outcome_price': '',
               'price_points': 0, 'news_items': 0, 'news_result_capped': False,
               'price_error': '', 'news_error': ''})
        row['topic'] = topic
        row['news_query'] = query
        if (slug in previous and not needs_deadline_window(row)
                and not row['price_error'] and (PRICES_ONLY or not row['news_error'])):
            rows.append(row)
            save_coverage(rows)
            print(f'{number:02d}/25 {slug} complete from previous run', flush=True)
            continue
        try:
            market = polymarket.market_by_slug(slug)
            row['question'] = market.question
            row['resolution_status'] = market.raw.get('umaResolutionStatus') or ''
            row['yes_outcome_price'] = yes_outcome_price(market.raw)
            if market.raw.get('closed') is not True or row['resolution_status'] != 'resolved':
                raise ValueError('Market is not confirmed closed and resolved')
            row['scheduled_end'] = market.raw['endDate']
            row['closed_time'] = market.raw.get('closedTime') or ''
            scheduled = parse_time(row['scheduled_end'])
            closed = parse_time(row['closed_time']) if row['closed_time'] else scheduled
            end = scheduled if closed - scheduled > timedelta(days=1) else closed
            start = max(parse_time(market.raw['startDate']), end - timedelta(days=7))
            window_changed = (slug in previous and
                              (row['window_start'] != iso(start) or row['window_end'] != iso(end)))
            if window_changed:
                with store.db:
                    store.db.execute('DELETE FROM prices WHERE market_slug = ?', (slug,))
                    store.db.execute('DELETE FROM news WHERE market_slug = ?', (slug,))
                row['price_points'] = 0
                row['news_items'] = 0
                row['news_result_capped'] = False
                row['price_error'] = 'pending'
                row['news_error'] = 'pending'
            row['window_start'] = iso(start)
            row['window_end'] = iso(end)
            store.put_market(market)
            if slug not in previous or row['price_error']:
                try:
                    prices = polymarket.prices(market, start, end, 'data-v2')
                    row['price_points'] = store.put_prices(prices)
                    row['price_error'] = ''
                except Exception as exc:
                    row['price_error'] = f'{type(exc).__name__}: {exc}'[:300]
            if PRICES_ONLY:
                if slug not in previous or row['news_error']:
                    row['news_error'] = 'pending'
            elif slug not in previous or row['news_error']:
                try:
                    news, capped = gdelt_sample(query, start, end, slug)
                    row['news_items'] = store.put_news(news)
                    row['news_result_capped'] = capped
                    row['news_error'] = ''
                except Exception as exc:
                    row['news_error'] = f'{type(exc).__name__}: {exc}'[:300]
        except Exception as exc:
            row['price_error'] = f'{type(exc).__name__}: {exc}'[:300]
        rows.append(row)
        print(f"{number:02d}/25 {slug} prices={row['price_points']} news={row['news_items']} price_error={row['price_error'][:60]} news_error={row['news_error'][:60]}", flush=True)
        save_coverage(rows)

    prices = [dict(x) for x in store.db.execute(
        'SELECT market_slug,timestamp,price,resolution_seconds,source FROM prices ORDER BY market_slug,timestamp')]
    news = [dict(x) for x in store.db.execute(
        'SELECT market_slug,source,query,url,title,published_at,provider_seen_at,first_collected_at FROM news ORDER BY market_slug,provider_seen_at,url')]
    write_csv(ROOT / 'prices.csv', ['market_slug','timestamp','price','resolution_seconds','source'], prices)
    write_csv(ROOT / 'news.csv', ['market_slug','source','query','url','title','published_at','provider_seen_at','first_collected_at'], news)
print(f'done markets={len(rows)} prices={len(prices)} news={len(news)}', flush=True)
