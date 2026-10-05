"""
Demonstration Datasets Generator for PRJ-07.
Provides realistic customer datasets (customer_a.csv and customer_b.csv)
with heterogeneous headers, formatting variations, cross-dataset overlapping entities,
and singletons.
"""

import os
import pandas as pd
from typing import Dict


def generate_demonstration_suite(output_dir: str = "data/demo") -> Dict[str, str]:
    """
    Ensure customer_a.csv and customer_b.csv exist and return their paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    path_a = os.path.join(output_dir, "customer_a.csv")
    path_b = os.path.join(output_dir, "customer_b.csv")

    if not os.path.exists(path_a):
        df_a = pd.DataFrame([
            {"first_name": "Rohan", "last_name": "Sharma", "email": "rohan.sharma@gmail.com", "phone_number": "+91 9876543210", "aadhaar": "4521 6789 1234", "city": "Mumbai", "address": "12 MG Road, Andheri"},
            {"first_name": "Priya", "last_name": "Patil", "email": "priya.patil@gmail.com", "phone_number": "+91 9988776655", "aadhaar": "5234 1122 3344", "city": "Pune", "address": "45 FC Road, Shivajinagar"},
            {"first_name": "Amit", "last_name": "Deshmukh", "email": "amit.d@gmail.com", "phone_number": "+91 9123456789", "aadhaar": "6789 2233 4455", "city": "Nashik", "address": "18 College Road, Nashik"},
        ])
        df_a.to_csv(path_a, index=False)

    if not os.path.exists(path_b):
        df_b = pd.DataFrame([
            {"full_name": "Rohan Kumar Sharma", "email_id": "rohan.sharma@gmail.com", "mobile_number": "9876543210", "aadhar_number": "452167891234", "city_name": "Mumbai", "residential_address": "12, M.G. Road, Andheri East"},
            {"full_name": "Priya N Patil", "email_id": "priya.patil@gmail.com", "mobile_number": "9988776655", "aadhar_number": "523411223344", "city_name": "Pune", "residential_address": "45 FC Road, Shivaji Nagar"},
            {"full_name": "Sneha Kulkarni", "email_id": "sneha.k@gmail.com", "mobile_number": "8899001122", "aadhar_number": "789012345678", "city_name": "Thane", "residential_address": "22 Ghodbunder Road, Thane"},
        ])
        df_b.to_csv(path_b, index=False)

    generated_files["customer_a"] = path_a
    generated_files["customer_b"] = path_b
    return generated_files
