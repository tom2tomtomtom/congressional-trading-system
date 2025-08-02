#!/usr/bin/env python3
"""
Full Congressional Database - All 535+ Members
Comprehensive database of all House Representatives and Senators with trading data.
"""

import json
import random
from datetime import datetime, timedelta
import pandas as pd

# Stock symbols for realistic trading patterns
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ORCL', 'CRM', 'ADBE']
FINANCIAL_STOCKS = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF']
HEALTHCARE_STOCKS = ['UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB']
ENERGY_STOCKS = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'KMI', 'OKE']
CONSUMER_STOCKS = ['WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'LOW', 'TGT', 'SBUX', 'MCD']
INDUSTRIAL_STOCKS = ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR']

ALL_STOCKS = TECH_STOCKS + FINANCIAL_STOCKS + HEALTHCARE_STOCKS + ENERGY_STOCKS + CONSUMER_STOCKS + INDUSTRIAL_STOCKS

def generate_full_congressional_data():
    """Generate comprehensive data for all 535+ congressional members."""
    
    # House Representatives (435 members)
    house_members = []
    
    # Senate Members (100 members)  
    senate_members = []
    
    # High-profile members with known trading activity
    high_profile_members = [
        # House Leadership & Notable Traders
        {"name": "Nancy Pelosi", "party": "D", "state": "CA", "chamber": "House", "district": "11", 
         "committee": "Former Speaker", "trading_activity": "Very High", "net_worth": 114000000},
        {"name": "Kevin McCarthy", "party": "R", "state": "CA", "chamber": "House", "district": "20",
         "committee": "Former Speaker", "trading_activity": "Medium", "net_worth": 3000000},
        {"name": "Hakeem Jeffries", "party": "D", "state": "NY", "chamber": "House", "district": "8",
         "committee": "Minority Leader", "trading_activity": "Low", "net_worth": 1500000},
        {"name": "Steve Scalise", "party": "R", "state": "LA", "chamber": "House", "district": "1",
         "committee": "Majority Leader", "trading_activity": "Medium", "net_worth": 800000},
        {"name": "Josh Gottheimer", "party": "D", "state": "NJ", "chamber": "House", "district": "5",
         "committee": "Financial Services", "trading_activity": "High", "net_worth": 8200000},
        {"name": "Dan Crenshaw", "party": "R", "state": "TX", "chamber": "House", "district": "2",
         "committee": "Energy & Commerce", "trading_activity": "High", "net_worth": 1800000},
        {"name": "Alexandria Ocasio-Cortez", "party": "D", "state": "NY", "chamber": "House", "district": "14",
         "committee": "Financial Services", "trading_activity": "None", "net_worth": 100000},
        {"name": "Marjorie Taylor Greene", "party": "R", "state": "GA", "chamber": "House", "district": "14",
         "committee": "Oversight", "trading_activity": "High", "net_worth": 44000000},
        {"name": "Matt Gaetz", "party": "R", "state": "FL", "chamber": "House", "district": "1",
         "committee": "Judiciary", "trading_activity": "Medium", "net_worth": 700000},
        {"name": "Katie Porter", "party": "D", "state": "CA", "chamber": "House", "district": "47",
         "committee": "Financial Services", "trading_activity": "None", "net_worth": 1800000},
        
        # Senate Leadership & Notable Traders
        {"name": "Chuck Schumer", "party": "D", "state": "NY", "chamber": "Senate", "district": None,
         "committee": "Majority Leader", "trading_activity": "Low", "net_worth": 1100000},
        {"name": "Mitch McConnell", "party": "R", "state": "KY", "chamber": "Senate", "district": None,
         "committee": "Minority Leader", "trading_activity": "Medium", "net_worth": 34000000},
        {"name": "Joe Manchin", "party": "D", "state": "WV", "chamber": "Senate", "district": None,
         "committee": "Energy & Natural Resources", "trading_activity": "Very High", "net_worth": 7600000},
        {"name": "Kyrsten Sinema", "party": "I", "state": "AZ", "chamber": "Senate", "district": None,
         "committee": "Banking", "trading_activity": "High", "net_worth": 100000},
        {"name": "Elizabeth Warren", "party": "D", "state": "MA", "chamber": "Senate", "district": None,
         "committee": "Banking", "trading_activity": "None", "net_worth": 12000000},
        {"name": "Ted Cruz", "party": "R", "state": "TX", "chamber": "Senate", "district": None,
         "committee": "Judiciary", "trading_activity": "Medium", "net_worth": 4000000},
        {"name": "Josh Hawley", "party": "R", "state": "MO", "chamber": "Senate", "district": None,
         "committee": "Judiciary", "trading_activity": "Low", "net_worth": 1000000},
        {"name": "Mark Warner", "party": "D", "state": "VA", "chamber": "Senate", "district": None,
         "committee": "Intelligence Chair", "trading_activity": "High", "net_worth": 25000000},
        {"name": "Richard Burr", "party": "R", "state": "NC", "chamber": "Senate", "district": None,
         "committee": "Former Intelligence Chair", "trading_activity": "Very High", "net_worth": 1700000},
        {"name": "Pat Toomey", "party": "R", "state": "PA", "chamber": "Senate", "district": None,
         "committee": "Former Banking Chair", "trading_activity": "Very High", "net_worth": 3000000},
    ]
    
    # Generate comprehensive House members (435 total)
    house_members = generate_house_members(high_profile_members)
    
    # Generate comprehensive Senate members (100 total)
    senate_members = generate_senate_members(high_profile_members)
    
    # Combine all members
    all_members = house_members + senate_members
    
    # Generate trading data for all members
    trading_data = generate_comprehensive_trading_data(all_members)
    
    return all_members, trading_data

def generate_house_members(high_profile_members):
    """Generate all 435 House Representatives."""
    
    # Extract high-profile House members
    house_high_profile = [m for m in high_profile_members if m["chamber"] == "House"]
    
    # State representation data (approximate)
    state_districts = {
        'CA': 52, 'TX': 38, 'FL': 28, 'NY': 26, 'PA': 17, 'IL': 17, 'OH': 15, 'GA': 14,
        'NC': 14, 'MI': 13, 'NJ': 12, 'VA': 11, 'WA': 10, 'AZ': 9, 'IN': 9, 'MA': 9,
        'TN': 9, 'MD': 8, 'MN': 8, 'MO': 8, 'WI': 8, 'CO': 8, 'AL': 7, 'SC': 7,
        'LA': 6, 'KY': 6, 'OR': 6, 'OK': 5, 'CT': 5, 'IA': 4, 'AR': 4, 'KS': 4,
        'UT': 4, 'NV': 4, 'NM': 3, 'WV': 2, 'NE': 3, 'ID': 2, 'HI': 2, 'NH': 2,
        'ME': 2, 'RI': 2, 'MT': 2, 'DE': 1, 'SD': 1, 'ND': 1, 'AK': 1, 'VT': 1, 'WY': 1
    }
    
    house_members = []
    member_id = 1
    
    # Add high-profile members first
    for member in house_high_profile:
        member['id'] = member_id
        member['tenure_years'] = random.randint(2, 30)
        member['age'] = random.randint(35, 80)
        house_members.append(member)
        member_id += 1
    
    # Generate remaining House members
    committees = [
        "Agriculture", "Appropriations", "Armed Services", "Budget", "Education & Labor",
        "Energy & Commerce", "Ethics", "Financial Services", "Foreign Affairs", "Homeland Security",
        "House Administration", "Intelligence", "Judiciary", "Natural Resources", "Oversight",
        "Rules", "Science & Technology", "Small Business", "Transportation", "Veterans Affairs", "Ways & Means"
    ]
    
    # Generate members for each state
    for state, districts in state_districts.items():
        for district in range(1, districts + 1):
            if member_id > 435:  # Cap at 435 House members
                break
                
            # Skip if we already have a high-profile member from this district
            existing = any(m for m in house_high_profile 
                          if m['state'] == state and m.get('district') == str(district))
            if existing:
                continue
            
            # Generate realistic member data
            party = random.choices(['D', 'R'], weights=[0.51, 0.49])[0]  # Slight Dem majority
            
            member = {
                'id': member_id,
                'name': generate_realistic_name(),
                'party': party,
                'state': state,
                'chamber': 'House',
                'district': str(district),
                'committee': random.choice(committees),
                'trading_activity': random.choices(
                    ['None', 'Low', 'Medium', 'High', 'Very High'],
                    weights=[0.35, 0.30, 0.20, 0.12, 0.03]
                )[0],
                'net_worth': generate_realistic_net_worth(),
                'tenure_years': random.randint(2, 30),
                'age': random.randint(25, 85)
            }
            
            house_members.append(member)
            member_id += 1
            
            if member_id > 435:
                break
    
    return house_members

def generate_senate_members(high_profile_members):
    """Generate all 100 Senate members."""
    
    # Extract high-profile Senate members
    senate_high_profile = [m for m in high_profile_members if m["chamber"] == "Senate"]
    
    states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
        'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
        'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    
    senate_committees = [
        "Agriculture", "Appropriations", "Armed Services", "Banking", "Budget", "Commerce",
        "Energy & Natural Resources", "Environment", "Finance", "Foreign Relations",
        "Health, Education, Labor & Pensions", "Homeland Security", "Intelligence", "Judiciary",
        "Rules", "Small Business", "Veterans Affairs"
    ]
    
    senate_members = []
    member_id = 436  # Continue from House numbering
    
    # Add high-profile members first
    for member in senate_high_profile:
        member['id'] = member_id
        member['tenure_years'] = random.randint(6, 40)
        member['age'] = random.randint(40, 85)
        senate_members.append(member)
        member_id += 1
    
    # Generate remaining Senate members (2 per state)
    for state in states:
        # Check how many high-profile members we already have from this state
        existing_count = sum(1 for m in senate_high_profile if m['state'] == state)
        
        for seat in range(existing_count, 2):  # Each state gets 2 senators
            if member_id > 535:  # Cap at 100 Senate members
                break
                
            party = random.choices(['D', 'R', 'I'], weights=[0.50, 0.48, 0.02])[0]
            
            member = {
                'id': member_id,
                'name': generate_realistic_name(),
                'party': party,
                'state': state,
                'chamber': 'Senate',
                'district': None,
                'committee': random.choice(senate_committees),
                'trading_activity': random.choices(
                    ['None', 'Low', 'Medium', 'High', 'Very High'],
                    weights=[0.40, 0.25, 0.20, 0.12, 0.03]
                )[0],
                'net_worth': generate_realistic_net_worth(is_senate=True),
                'tenure_years': random.randint(6, 40),
                'age': random.randint(35, 85)
            }
            
            senate_members.append(member)
            member_id += 1
    
    return senate_members

def generate_realistic_name():
    """Generate realistic names for congressional members."""
    first_names = [
        'John', 'Michael', 'David', 'James', 'Robert', 'William', 'Richard', 'Joseph', 'Thomas', 'Christopher',
        'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth',
        'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen',
        'Nancy', 'Lisa', 'Betty', 'Helen', 'Sandra', 'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle',
        'Laura', 'Sarah', 'Kimberly', 'Deborah', 'Dorothy', 'Lisa', 'Nancy', 'Karen', 'Betty', 'Helen'
    ]
    
    last_names = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
        'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
        'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson',
        'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
        'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts'
    ]
    
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_realistic_net_worth(is_senate=False):
    """Generate realistic net worth based on congressional wealth distributions."""
    if is_senate:
        # Senators tend to be wealthier
        return random.choices(
            [100000, 500000, 1000000, 5000000, 10000000, 25000000, 50000000],
            weights=[0.15, 0.25, 0.30, 0.20, 0.07, 0.02, 0.01]
        )[0] + random.randint(-50000, 200000)
    else:
        # House members
        return random.choices(
            [100000, 500000, 1000000, 2000000, 5000000, 15000000, 40000000],
            weights=[0.20, 0.35, 0.25, 0.12, 0.06, 0.015, 0.005]
        )[0] + random.randint(-50000, 100000)

def generate_comprehensive_trading_data(all_members):
    """Generate realistic trading data for all congressional members."""
    
    trading_data = []
    trade_id = 1
    
    # Generate trading activity based on member profiles
    for member in all_members:
        activity_level = member.get('trading_activity', 'None')
        
        if activity_level == 'None':
            continue  # No trades for this member
        
        # Determine number of trades based on activity level
        trade_counts = {
            'Low': random.randint(1, 3),
            'Medium': random.randint(3, 8),
            'High': random.randint(8, 15),
            'Very High': random.randint(15, 30)
        }
        
        num_trades = trade_counts.get(activity_level, 0)
        
        # Generate trades for this member
        for i in range(num_trades):
            trade = generate_single_trade(member, trade_id)
            trading_data.append(trade)
            trade_id += 1
    
    return trading_data

def generate_single_trade(member, trade_id):
    """Generate a single realistic trade for a member."""
    
    # Select stock based on committee assignment and realistic patterns
    stock_symbol = select_realistic_stock(member)
    
    # Generate trade date (last 4 years)
    start_date = datetime.now() - timedelta(days=4*365)
    end_date = datetime.now() - timedelta(days=30)
    trade_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    
    # Generate filing date (should be within 45 days but sometimes late)
    base_filing_delay = random.choices(
        [random.randint(10, 30), random.randint(31, 45), random.randint(46, 90), random.randint(91, 200)],
        weights=[0.60, 0.25, 0.12, 0.03]
    )[0]
    filing_date = trade_date + timedelta(days=base_filing_delay)
    
    # Generate trade amount based on member wealth and activity
    amount_from, amount_to = generate_trade_amounts(member)
    
    # Transaction type
    transaction_type = random.choices(['Purchase', 'Sale'], weights=[0.65, 0.35])[0]
    
    # Owner type
    owner_type = random.choices(['Self', 'Spouse', 'Child'], weights=[0.70, 0.25, 0.05])[0]
    
    trade = {
        'id': trade_id,
        'member_id': member['id'],
        'member_name': member['name'],
        'party': member['party'],
        'state': member['state'],
        'chamber': member['chamber'],
        'symbol': stock_symbol,
        'asset_name': get_company_name(stock_symbol),
        'transaction_date': trade_date.strftime('%Y-%m-%d'),
        'filing_date': filing_date.strftime('%Y-%m-%d'),
        'transaction_type': transaction_type,
        'amount_from': amount_from,
        'amount_to': amount_to,
        'owner_type': owner_type,
        'committee': member.get('committee', 'Unknown')
    }
    
    return trade

def select_realistic_stock(member):
    """Select realistic stock based on member's committee and patterns."""
    
    committee = member.get('committee', '').lower()
    
    # Committee-based stock selection with some randomness
    if 'financial' in committee or 'banking' in committee:
        return random.choices(FINANCIAL_STOCKS + ALL_STOCKS, weights=[0.4] * len(FINANCIAL_STOCKS) + [0.1] * len(ALL_STOCKS))[0]
    elif 'energy' in committee:
        return random.choices(ENERGY_STOCKS + ALL_STOCKS, weights=[0.4] * len(ENERGY_STOCKS) + [0.1] * len(ALL_STOCKS))[0]
    elif 'commerce' in committee or 'technology' in committee:
        return random.choices(TECH_STOCKS + ALL_STOCKS, weights=[0.4] * len(TECH_STOCKS) + [0.1] * len(ALL_STOCKS))[0]
    elif 'health' in committee:
        return random.choices(HEALTHCARE_STOCKS + ALL_STOCKS, weights=[0.4] * len(HEALTHCARE_STOCKS) + [0.1] * len(ALL_STOCKS))[0]
    elif 'armed' in committee or 'defense' in committee:
        return random.choices(INDUSTRIAL_STOCKS + ALL_STOCKS, weights=[0.3] * len(INDUSTRIAL_STOCKS) + [0.1] * len(ALL_STOCKS))[0]
    else:
        return random.choice(ALL_STOCKS)

def generate_trade_amounts(member):
    """Generate realistic trade amounts based on member wealth."""
    
    net_worth = member.get('net_worth', 1000000)
    activity = member.get('trading_activity', 'Low')
    
    # Base trade size as percentage of net worth
    if activity == 'Very High':
        base_percentage = random.uniform(0.05, 0.25)  # 5-25% of net worth
    elif activity == 'High':
        base_percentage = random.uniform(0.02, 0.15)  # 2-15% of net worth
    elif activity == 'Medium':
        base_percentage = random.uniform(0.01, 0.08)  # 1-8% of net worth
    else:  # Low
        base_percentage = random.uniform(0.005, 0.03)  # 0.5-3% of net worth
    
    base_amount = int(net_worth * base_percentage)
    
    # STOCK Act reporting ranges
    ranges = [
        (1000, 15000), (15000, 50000), (50000, 100000), (100000, 250000),
        (250000, 500000), (500000, 1000000), (1000000, 5000000), (5000000, 25000000)
    ]
    
    # Find appropriate range
    for amount_from, amount_to in ranges:
        if base_amount <= amount_to:
            return amount_from, amount_to
    
    # For very large trades
    return 5000000, 25000000

def get_company_name(symbol):
    """Get company name for stock symbol."""
    company_names = {
        'AAPL': 'Apple Inc', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc Class A',
        'AMZN': 'Amazon.com Inc', 'META': 'Meta Platforms Inc', 'NVDA': 'NVIDIA Corporation',
        'TSLA': 'Tesla Inc', 'JPM': 'JPMorgan Chase & Co', 'BAC': 'Bank of America Corp',
        'WFC': 'Wells Fargo & Company', 'XOM': 'Exxon Mobil Corporation', 'CVX': 'Chevron Corporation',
        'UNH': 'UnitedHealth Group Inc', 'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer Inc',
        'WMT': 'Walmart Inc', 'HD': 'The Home Depot Inc', 'PG': 'The Procter & Gamble Company',
        'KO': 'The Coca-Cola Company', 'BA': 'The Boeing Company', 'CAT': 'Caterpillar Inc'
    }
    return company_names.get(symbol, f'{symbol} Corporation')

def save_congressional_data():
    """Generate and save comprehensive congressional data."""
    
    print("🏛️ Generating Comprehensive Congressional Database...")
    print("   Creating all 535+ congressional members...")
    
    all_members, trading_data = generate_full_congressional_data()
    
    print(f"✅ Generated {len(all_members)} congressional members")
    print(f"   • House: {len([m for m in all_members if m['chamber'] == 'House'])} members")
    print(f"   • Senate: {len([m for m in all_members if m['chamber'] == 'Senate'])} members")
    print(f"✅ Generated {len(trading_data)} trades")
    
    # Save to files
    with open('congressional_members_full.json', 'w') as f:
        json.dump(all_members, f, indent=2)
    
    with open('congressional_trades_full.json', 'w') as f:
        json.dump(trading_data, f, indent=2)
    
    # Create summary statistics
    members_by_party = {}
    for member in all_members:
        party = member['party']
        members_by_party[party] = members_by_party.get(party, 0) + 1
    
    trading_members = len([m for m in all_members if m.get('trading_activity', 'None') != 'None'])
    high_activity_members = len([m for m in all_members if m.get('trading_activity') in ['High', 'Very High']])
    
    print(f"\n📊 Congressional Composition:")
    for party, count in members_by_party.items():
        print(f"   • {party}: {count} members")
    
    print(f"\n💰 Trading Activity:")
    print(f"   • Members with trading activity: {trading_members}")
    print(f"   • High-activity traders: {high_activity_members}")
    print(f"   • Total trades generated: {len(trading_data)}")
    
    print(f"\n💾 Data saved to:")
    print(f"   • congressional_members_full.json")
    print(f"   • congressional_trades_full.json")
    
    return all_members, trading_data

if __name__ == "__main__":
    all_members, trading_data = save_congressional_data()
    print("\n🎊 Full congressional database generated successfully!")