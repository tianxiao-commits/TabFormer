"""
Generate synthetic credit card transaction dataset for TabFormer testing.
This creates a dataset with the same schema as the original credit card transaction data.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
import os


def generate_synthetic_transactions(num_rows=10000, num_users=100, seed=42):
    """
    Generate synthetic credit card transactions matching the TabFormer schema.

    Schema: User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
            Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating {num_rows} synthetic transactions for {num_users} users...")

    # Define some realistic values
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    cities = ['Los Angeles', 'New York', 'Houston', 'Phoenix', 'Philadelphia',
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin']
    merchant_names = [f'Merchant_{i:04d}' for i in range(500)]
    chip_options = ['Swipe Transaction', 'Chip Transaction', 'Online Transaction']
    error_types = ['None', 'Bad PIN', 'Bad CVV', 'Technical Glitch', None]
    mcc_codes = [5411, 5912, 5814, 5541, 5812, 5999, 7011, 4121, 5311, 5722]

    transactions = []
    start_date = datetime(2019, 1, 1)

    for _ in range(num_rows):
        # Generate timestamp
        random_days = random.randint(0, 730)  # 2 years
        random_seconds = random.randint(0, 86400)  # seconds in a day
        trans_date = start_date + timedelta(days=random_days, seconds=random_seconds)

        # Generate transaction
        transaction = {
            'User': random.randint(0, num_users - 1),
            'Card': random.randint(0, num_users * 2),  # Some users have multiple cards
            'Year': trans_date.year,
            'Month': trans_date.month,
            'Day': trans_date.day,
            'Time': trans_date.strftime('%H:%M'),
            'Amount': f'${random.uniform(1, 5000):.2f}',
            'Use Chip': random.choice(chip_options),
            'Merchant Name': random.choice(merchant_names),
            'Merchant City': random.choice(cities),
            'Merchant State': random.choice(states),
            'Zip': random.randint(10000, 99999),
            'MCC': random.choice(mcc_codes),
            'Errors?': random.choice(error_types) if random.random() < 0.1 else None,
            'Is Fraud?': 'Yes' if random.random() < 0.02 else 'No'  # 2% fraud rate
        }
        transactions.append(transaction)

    # Create DataFrame and sort by User and timestamp
    df = pd.DataFrame(transactions)
    df = df.sort_values(['User', 'Year', 'Month', 'Day', 'Time']).reset_index(drop=True)

    print(f"Generated dataset shape: {df.shape}")
    print(f"Number of unique users: {df['User'].nunique()}")
    print(f"Fraud rate: {(df['Is Fraud?'] == 'Yes').sum() / len(df) * 100:.2f}%")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic credit card transaction data')
    parser.add_argument('--nrows', type=int, default=10000,
                        help='Number of transaction rows to generate (default: 10000)')
    parser.add_argument('--nusers', type=int, default=100,
                        help='Number of unique users (default: 100)')
    parser.add_argument('--output', type=str, default='./data/credit_card/card_transaction.v1.csv',
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Generate data
    df = generate_synthetic_transactions(
        num_rows=args.nrows,
        num_users=args.nusers,
        seed=args.seed
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSynthetic dataset saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()
