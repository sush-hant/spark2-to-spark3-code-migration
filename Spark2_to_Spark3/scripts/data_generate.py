import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate dates for 2 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start_date, end_date, freq='D')

# Categories and subcategories
categories = {
    'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera', 'Smart Watch'],
    'Clothing': ['Shirt', 'Pants', 'Dress', 'Jacket', 'Shoes', 'Accessories'],
    'Home & Kitchen': ['Cookware', 'Furniture', 'Bedding', 'Decor', 'Appliances'],
    'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Comics', 'Magazine'],
    'Sports': ['Equipment', 'Clothing', 'Supplements', 'Accessories'],
    'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrance'],
    'Grocery': ['Snacks', 'Beverages', 'Organic', 'Dairy', 'Frozen']
}

# Regions and cities
regions = {
    'North': ['New York', 'Boston', 'Chicago', 'Detroit'],
    'South': ['Miami', 'Houston', 'Atlanta', 'Dallas'],
    'East': ['Philadelphia', 'Charlotte', 'Washington DC'],
    'West': ['Los Angeles', 'San Francisco', 'Seattle', 'Phoenix']
}

# Payment methods
payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Amazon Pay', 'Cash on Delivery']

# Shipping methods
shipping_methods = ['Standard', 'Express', 'Same Day', 'Prime']

# Customer segments
customer_segments = ['Premium', 'Regular', 'Occasional', 'New']

# Generate Orders
n_orders = 10000
orders_data = []

for i in range(n_orders):
    order_id = f'ORD{str(i+1).zfill(6)}'
    customer_id = f'CUST{random.randint(1, 2000):05d}'
    order_date = random.choice(date_range)
    
    # Select region and city
    region = random.choice(list(regions.keys()))
    city = random.choice(regions[region])
    
    # Select category and product
    category = random.choice(list(categories.keys()))
    product_type = random.choice(categories[category])
    product_id = f'PROD{random.randint(1, 5000):05d}'
    
    # Generate quantities and prices based on category
    if category == 'Electronics':
        unit_price = round(random.uniform(200, 2000), 2)
        quantity = random.randint(1, 3)
    elif category == 'Clothing':
        unit_price = round(random.uniform(15, 150), 2)
        quantity = random.randint(1, 5)
    elif category == 'Books':
        unit_price = round(random.uniform(10, 50), 2)
        quantity = random.randint(1, 4)
    elif category == 'Grocery':
        unit_price = round(random.uniform(5, 100), 2)
        quantity = random.randint(1, 10)
    else:
        unit_price = round(random.uniform(20, 500), 2)
        quantity = random.randint(1, 4)
    
    total_amount = round(unit_price * quantity, 2)
    
    # Discount (20% of orders have discount)
    discount_percent = random.choice([0, 0, 0, 0, 5, 10, 15, 20, 25])
    discount_amount = round(total_amount * discount_percent / 100, 2)
    final_amount = round(total_amount - discount_amount, 2)
    
    # Shipping and tax
    shipping_method = random.choice(shipping_methods)
    shipping_cost = 0 if shipping_method == 'Prime' else random.choice([0, 5, 10, 15])
    tax_amount = round(final_amount * 0.08, 2)
    grand_total = round(final_amount + shipping_cost + tax_amount, 2)
    
    # Payment and delivery
    payment_method = random.choice(payment_methods)
    customer_segment = random.choice(customer_segments)
    
    # Delivery date (1-7 days after order)
    delivery_days = {'Standard': 5, 'Express': 3, 'Same Day': 1, 'Prime': 2}
    delivery_date = order_date + timedelta(days=delivery_days[shipping_method])
    
    # Order status
    order_status = random.choices(
        ['Delivered', 'Shipped', 'Processing', 'Cancelled', 'Returned'],
        weights=[70, 15, 5, 5, 5]
    )[0]
    
    # Rating (only for delivered orders)
    rating = random.randint(1, 5) if order_status == 'Delivered' else None
    
    # Prime member
    is_prime = random.choice([True, False])
    
    orders_data.append({
        'order_id': order_id,
        'customer_id': customer_id,
        'order_date': order_date.strftime('%Y-%m-%d'),
        'delivery_date': delivery_date.strftime('%Y-%m-%d') if order_status != 'Cancelled' else None,
        'product_id': product_id,
        'product_type': product_type,
        'category': category,
        'quantity': quantity,
        'unit_price': unit_price,
        'total_amount': total_amount,
        'discount_percent': discount_percent,
        'discount_amount': discount_amount,
        'final_amount': final_amount,
        'shipping_method': shipping_method,
        'shipping_cost': shipping_cost,
        'tax_amount': tax_amount,
        'grand_total': grand_total,
        'payment_method': payment_method,
        'region': region,
        'city': city,
        'customer_segment': customer_segment,
        'order_status': order_status,
        'rating': rating,
        'is_prime_member': is_prime
    })

# Create DataFrame
df_orders = pd.DataFrame(orders_data)

# Generate Customer Demographics
customer_ids = df_orders['customer_id'].unique()
customer_demo_data = []

for cust_id in customer_ids:
    age = random.randint(18, 70)
    gender = random.choice(['M', 'F', 'Other'])
    account_age_days = random.randint(30, 1825)  # 1 month to 5 years
    total_lifetime_orders = random.randint(1, 50)
    
    customer_demo_data.append({
        'customer_id': cust_id,
        'age': age,
        'gender': gender,
        'account_age_days': account_age_days,
        'total_lifetime_orders': total_lifetime_orders,
        'preferred_payment': random.choice(payment_methods)
    })

df_customers = pd.DataFrame(customer_demo_data)

# Generate Product Catalog
product_ids = df_orders['product_id'].unique()
product_catalog_data = []

for prod_id in product_ids:
    brand = f'Brand_{random.choice(["A", "B", "C", "D", "E", "F", "G", "H"])}'
    seller_id = f'SELLER{random.randint(1, 100):03d}'
    stock_quantity = random.randint(0, 1000)
    avg_rating = round(random.uniform(2.5, 5.0), 1)
    num_reviews = random.randint(0, 5000)
    
    product_catalog_data.append({
        'product_id': prod_id,
        'brand': brand,
        'seller_id': seller_id,
        'stock_quantity': stock_quantity,
        'avg_rating': avg_rating,
        'num_reviews': num_reviews
    })

df_products = pd.DataFrame(product_catalog_data)

# Save to CSV
df_orders.to_csv('amazon_orders.csv', index=False)
df_customers.to_csv('amazon_customers.csv', index=False)
df_products.to_csv('amazon_products.csv', index=False)

print(f"Generated {len(df_orders)} orders")
print(f"Generated {len(df_customers)} customer records")
print(f"Generated {len(df_products)} product records")
print("\nFiles created:")
print("- amazon_orders.csv")
print("- amazon_customers.csv")
print("- amazon_products.csv")
print("\nSample aggregations possible:")
print("- Revenue by category, region, time period")
print("- Customer segmentation and lifetime value")
print("- Product performance and inventory analysis")
print("- Shipping method efficiency")
print("- Payment method preferences")
print("- Seasonal trends and patterns")
print("- Customer retention and churn")
print("- Rating analysis and correlations")