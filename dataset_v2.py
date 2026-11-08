import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

num_records = 210850  # Exactly 210,850 rows (~39.5 MB CSV file)

carriers = [
    "Blue Dart",
    "Delhivery",
    "Ekart",
    "XpressBees",
    "Ecom Express",
    "Shadowfax",
]
cities = [
    "Kolkata",
    "Delhi_NCR",
    "Mumbai",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Ahmedabad",
    "Guwahati",
    "Srinagar",
]
zones = [
    "Zone A (Intra-City)",
    "Zone B (Within State/Region)",
    "Zone C (Metro-to-Metro)",
    "Zone D (Rest of India)",
    "Zone E (NE / J&K / Special)",
]
shipment_modes = ["Surface", "Air"]
festivals = [
    "None",
    "Durga Puja / Dussehra",
    "Diwali Peak",
    "Eid-ul-Fitr",
    "Holi",
    "Independence Day Sale",
    "New Year Carnival",
]

pos_pool = [
    "Good product!",
    "Loved it!",
    "Fast delivery.",
    "Very nice.",
    "Awesome quality.",
    "Superb service.",
    "Delivered well within the SLA window without any issues.",
    (
        "Got my package safe and sound. The tracking info was updated promptly"
        " on the app."
    ),
    (
        "Extremely fast shipping, especially considering the heavy festive"
        " rush! Very satisfied with the courier partner."
    ),
]

neu_pool = [
    "It's okay.",
    "Average product.",
    "Took a bit long.",
    "Fine.",
    "Okay quality.",
    "Mediocre service.",
    (
        "Delivery took a little longer than expected due to local traffic"
        " congestion, but the item arrived safely."
    ),
    (
        "The outer cardboard box was slightly crushed on one corner, though"
        " the inner product was completely undamaged."
    ),
    (
        "Average experience with courier tracking. Updates on the app were"
        " delayed by a few hours."
    ),
]

neg_pool = [
    "Bad product.",
    "Terrible service.",
    "Late delivery.",
    "Damaged box.",
    "Worst experience.",
    "Not good.",
    (
        "Terrible experience! Delayed by several days without any prior"
        " notification or status update."
    ),
    (
        "The package arrived completely torn and damp. Very disappointed with"
        " the transit handling."
    ),
    (
        "Delivery executive was extremely rude and asked me to come pick it up"
        " from the main road."
    ),
]

print(f"Generating {num_records:,} shipment records for Version 2...")

output_filename = "india_multi_agency_edd_dataset_v2.csv"

# Generate base features
order_ids = [f"IND2026{str(j).zfill(8)}" for j in range(num_records)]
delivery_partners = np.random.choice(carriers, size=num_records)
origin_city = np.random.choice(cities, size=num_records)
destination_city = np.random.choice(cities, size=num_records)
origin_pincode = np.random.randint(110001, 800000, size=num_records)
destination_pincode = np.random.randint(110001, 800000, size=num_records)
logistics_zone = np.random.choice(zones, size=num_records)
distance_km = np.round(np.random.uniform(5.0, 2500.0, size=num_records), 2)
shipment_mode = np.random.choice(
    shipment_modes, size=num_records, p=[0.75, 0.25]
)
weight_kg = np.round(np.random.uniform(0.1, 25.0, size=num_records), 2)

# Restoring Version 1 cost & operational columns
warehouse_handling_fee = np.round(
    np.random.uniform(10.0, 50.0, size=num_records), 2
)
fuel_surcharge = np.round(
    weight_kg * np.random.uniform(5.0, 15.0, size=num_records), 2
)
freight_cost = np.round(
    distance_km * np.where(shipment_mode == "Air", 2.5, 1.2)
    + warehouse_handling_fee
    + fuel_surcharge,
    2,
)
insurance_fee = np.round(freight_cost * 0.05, 2)
total_shipping_cost = np.round(freight_cost + insurance_fee, 2)

# Seasonal & Delivery Performance calculations
festival_peak_col = np.random.choice(
    festivals, size=num_records, p=[0.70, 0.08, 0.10, 0.03, 0.04, 0.02, 0.03]
)
base_hours = distance_km / np.where(
    shipment_mode == "Air", 65.0, 28.0
) + np.random.gamma(2, 4, size=num_records)
festival_delay = np.where(
    festival_peak_col != "None", np.random.uniform(18, 60, size=num_records), 0
)
transit_time_hours = np.round(base_hours + festival_delay, 2)
actual_delivery_days = np.round(transit_time_hours / 24.0, 2)
is_delayed_sla = np.where(
    transit_time_hours > np.where(shipment_mode == "Air", 72, 144), 1, 0
)

# Ratings & Reviews with synthetic NaNs
ratings = np.where(
    is_delayed_sla == 1,
    np.random.choice([1, 2, 3], size=num_records, p=[0.65, 0.25, 0.10]),
    np.random.choice([4, 5], size=num_records, p=[0.25, 0.75]),
).astype(float)
nan_rating_mask = np.random.rand(num_records) < 0.03
ratings[nan_rating_mask] = np.nan

reviews = []
for r in ratings:
  if pd.isna(r):
    reviews.append(np.nan)
  elif r >= 4:
    reviews.append(np.random.choice(pos_pool))
  elif r == 3:
    reviews.append(np.random.choice(neu_pool))
  else:
    reviews.append(np.random.choice(neg_pool))

nan_review_mask = np.random.rand(num_records) < 0.02
for idx in range(num_records):
  if nan_review_mask[idx] and not pd.isna(ratings[idx]):
    reviews[idx] = np.nan

# Build comprehensive DataFrame combining Version 1 and Version 2 fields
df = pd.DataFrame({
    "Order_ID": order_ids,
    "Delivery_Partner": delivery_partners,
    "Origin_City": origin_city,
    "Origin_Pincode": origin_pincode,
    "Destination_City": destination_city,
    "Destination_Pincode": destination_pincode,
    "Logistics_Zone": logistics_zone,
    "Distance_km": distance_km,
    "Shipment_Mode": shipment_mode,
    "Weight_kg": weight_kg,
    "Warehouse_Handling_Fee": warehouse_handling_fee,
    "Fuel_Surcharge": fuel_surcharge,
    "Freight_Cost": freight_cost,
    "Insurance_Fee": insurance_fee,
    "Total_Shipping_Cost": total_shipping_cost,
    "Festival_Peak": festival_peak_col,
    "Transit_Time_Hours": transit_time_hours,
    "Actual_Delivery_Days": actual_delivery_days,
    "Is_Delayed_SLA": is_delayed_sla,
    "Rating_Stars": ratings,
    "Review_Text": reviews,
})

df.to_csv(output_filename, index=False)
print(
    f"Dataset generated successfully! Saved as '{output_filename}' with"
    f" {len(df):,} rows and {len(df.columns)} columns."
)