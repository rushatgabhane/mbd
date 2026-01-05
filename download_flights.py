#!/usr/bin/env python3
"""
Automated download of BTS On-Time Flight Performance data.

Downloads all columns for years 1987-2025, all months.
Handles ASP.NET ViewState tokens properly.
"""

import requests
import zipfile
import io
import os
import time
import sys
import re
from html.parser import HTMLParser


# Output directory
OUTPUT_DIR = './flight_data/'

# All available fields
FIELDS = [
    'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE',
    'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM',
    'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN',
    'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC',
    'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST',
    'DEST_CITY_NAME', 'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC',
    'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP',
    'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN',
    'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP',
    'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED',
    'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP',
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
    'FIRST_DEP_TIME', 'TOTAL_ADD_GTIME', 'LONGEST_ADD_GTIME',
    'DIV_AIRPORT_LANDINGS', 'DIV_REACHED_DEST', 'DIV_ACTUAL_ELAPSED_TIME', 'DIV_ARR_DELAY', 'DIV_DISTANCE',
    'DIV1_AIRPORT', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID', 'DIV1_WHEELS_ON',
    'DIV1_TOTAL_GTIME', 'DIV1_LONGEST_GTIME', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM'
]

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

FORM_URL = 'https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr'


def extract_hidden_fields(html):
    """Extract ASP.NET hidden fields from HTML."""
    fields = {}

    # Extract __VIEWSTATE
    match = re.search(r'id="__VIEWSTATE"\s+value="([^"]*)"', html)
    if match:
        fields['__VIEWSTATE'] = match.group(1)

    # Extract __VIEWSTATEGENERATOR
    match = re.search(r'id="__VIEWSTATEGENERATOR"\s+value="([^"]*)"', html)
    if match:
        fields['__VIEWSTATEGENERATOR'] = match.group(1)

    # Extract __EVENTVALIDATION
    match = re.search(r'id="__EVENTVALIDATION"\s+value="([^"]*)"', html)
    if match:
        fields['__EVENTVALIDATION'] = match.group(1)

    return fields


def get_form_page(session):
    """Fetch the form page and extract hidden fields."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    response = session.get(FORM_URL, headers=headers, timeout=60)
    response.raise_for_status()

    return extract_hidden_fields(response.text)


def download_month(session, year, month, hidden_fields):
    """
    Download flight data for a specific year and month.
    """
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.transtats.bts.gov',
        'Referer': FORM_URL,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    month_name = MONTHS[month - 1]
    output_file = os.path.join(OUTPUT_DIR, f'{year}-{month:02d}.csv')

    # Skip if already downloaded
    if os.path.exists(output_file):
        print(f'SKIP: {month_name} {year} (already exists)')
        return True, 'skip'

    # Build form data
    data = {
        '__EVENTTARGET': '',
        '__EVENTARGUMENT': '',
        '__LASTFOCUS': '',
        '__VIEWSTATE': hidden_fields.get('__VIEWSTATE', ''),
        '__VIEWSTATEGENERATOR': hidden_fields.get('__VIEWSTATEGENERATOR', ''),
        '__EVENTVALIDATION': hidden_fields.get('__EVENTVALIDATION', ''),
        'txtSearch': '',
        'cboGeography': 'All',
        'cboYear': str(year),
        'cboPeriod': str(month),
        'btnDownload': 'Download',
    }

    # Add all field checkboxes
    for field in FIELDS:
        data[field] = 'on'

    try:
        response = session.post(FORM_URL, headers=headers, data=data, timeout=300)
        response.raise_for_status()

        # Check if we got a zip file (starts with PK)
        if len(response.content) < 100:
            print(f'FAIL: {month_name} {year} - Empty response')
            return False, 'fail'

        if response.content[:2] != b'PK':
            # Might be HTML error page, try to get new tokens
            print(f'WARN: {month_name} {year} - Got HTML instead of ZIP, refreshing tokens...')
            return False, 'refresh'

        # Extract the zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # Find and extract the CSV
        for filename in z.namelist():
            if filename.endswith('.csv'):
                z.extract(filename, OUTPUT_DIR)
                old_path = os.path.join(OUTPUT_DIR, filename)
                os.rename(old_path, output_file)
                break

        print(f'OK: {month_name} {year}')
        return True, 'ok'

    except zipfile.BadZipFile:
        print(f'FAIL: {month_name} {year} - Bad zip file')
        return False, 'fail'
    except requests.exceptions.Timeout:
        print(f'FAIL: {month_name} {year} - Timeout')
        return False, 'fail'
    except requests.exceptions.RequestException as e:
        print(f'FAIL: {month_name} {year} - {e}')
        return False, 'fail'
    except Exception as e:
        print(f'FAIL: {month_name} {year} - {e}')
        return False, 'fail'


def main():
    """Main function to download all flight data."""

    start_year = 1987
    end_year = 2025

    if len(sys.argv) >= 3:
        start_year = int(sys.argv[1])
        end_year = int(sys.argv[2])
        print(f'Custom range: {start_year} to {end_year}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = requests.Session()

    # Statistics
    success_count = 0
    fail_count = 0
    skip_count = 0

    total_months = (end_year - start_year + 1) * 12
    current = 0

    print(f'Starting download: {start_year}-01 to {end_year}-12')
    print(f'Output directory: {os.path.abspath(OUTPUT_DIR)}')
    print(f'Total months to process: {total_months}')
    print('-' * 50)

    # Get initial form tokens
    print('Fetching form tokens...')
    try:
        hidden_fields = get_form_page(session)
        print(f'Got ViewState ({len(hidden_fields.get("__VIEWSTATE", ""))} chars)')
    except Exception as e:
        print(f'ERROR: Could not fetch form page: {e}')
        return

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            current += 1

            # For 2025, data might not be available for later months
            if year == 2025 and month > 10:
                print(f'[{current}/{total_months}] SKIP: {MONTHS[month-1]} {year} (future)')
                skip_count += 1
                continue

            output_file = os.path.join(OUTPUT_DIR, f'{year}-{month:02d}.csv')
            if os.path.exists(output_file):
                print(f'[{current}/{total_months}] SKIP: {MONTHS[month-1]} {year} (exists)')
                skip_count += 1
                continue

            # Get fresh tokens for each download
            try:
                hidden_fields = get_form_page(session)
            except Exception as e:
                print(f'[{current}/{total_months}] FAIL: {MONTHS[month-1]} {year} - Could not get tokens: {e}')
                fail_count += 1
                time.sleep(5)
                continue

            print(f'[{current}/{total_months}] ', end='')

            success, status = download_month(session, year, month, hidden_fields)

            if status == 'skip':
                skip_count += 1
            elif status == 'ok':
                success_count += 1
            else:
                fail_count += 1

            # Rate limiting
            time.sleep(2)

    print('-' * 50)
    print(f'Complete! Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}')
    print(f'Files saved to: {os.path.abspath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()
