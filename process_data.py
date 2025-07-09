import pandas as pd
import os

# --- Configuration ---
INPUT_CSV_FILE = 'raw_customer_orders.csv'  # Your downloaded Google Sheet CSV
OUTPUT_CSV_FILE = 'customer_orders.csv'     # The file our main app.py will use
DELIMITER = '/' # The character used to separate customer phrases in your Google Sheet's "Customer Order" column

def process_customer_data(input_file, output_file, delimiter):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found. Please ensure your downloaded Google Sheet CSV is in the same directory.")
        return

    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows from {input_file}.")
    except Exception as e:
        print(f"Error reading CSV file. Please check if it's a valid CSV and the path is correct. Error: {e}")
        return

    # --- IMPORTANT: Rename columns to match your Google Sheet headers exactly ---
    df = df.rename(columns={
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
        'HKM Item Code': 'internal_sku',
        'HKM Description': 'hkm_description',
        'Customer Order': 'customer_order_raw'
    })

    # Ensure essential columns exist after renaming
    required_columns = ['customer_id', 'customer_name', 'internal_sku', 'customer_order_raw']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more required columns after renaming.")
        print(f"Please ensure your Google Sheet has these headers (and they are correctly typed and cased in the script):")
        print(f"'{', '.join(required_columns)}'")
        print(f"Found columns in your CSV: {df.columns.tolist()}")
        return

    processed_rows = []
    print("Processing customer order phrases...")
    for index, row in df.iterrows():
        customer_id = row['customer_id']
        customer_name = row['customer_name']
        internal_sku = row['internal_sku']
        hkm_description = row['hkm_description']
        customer_order_raw = str(row['customer_order_raw']).strip()

        phrases = [p.strip() for p in customer_order_raw.split(delimiter) if p.strip()]

        if not phrases:
            print(f"Warning: No valid customer phrases found for SKU {internal_sku} (Row {index}). Skipping.")
            continue

        for phrase in phrases:
            processed_rows.append({
                'customer_id': customer_id,
                'customer_name': customer_name,
                'customer_phrase': phrase,
                'internal_sku': internal_sku,
                'hkm_description': hkm_description
            })

    df_processed = pd.DataFrame(processed_rows)
    final_columns = ['customer_id', 'customer_name', 'customer_phrase', 'internal_sku']
    df_final = df_processed[final_columns]
    df_final.to_csv(output_file, index=False)
    print(f"Successfully processed data and saved to {output_file}. Total {len(df_final)} entries.")
    print("\n--- Next Step ---")
    print(f"You can now run your main Streamlit app by typing: streamlit run app.py")


if __name__ == "__main__":
    process_customer_data(INPUT_CSV_FILE, OUTPUT_CSV_FILE, DELIMITER)
