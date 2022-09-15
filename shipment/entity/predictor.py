import os
import sys

from shipment.exception import ShipmentException
from shipment.utils.util import load_object

import pandas as pd


class ShipmentData:

    def __init__(self,
                 country: str,
                 Managed_By: str,
                 Fulfill_Via: str,
                 Vendor_INCO_Term: str,
                 Shipment_Mode: str,
                 Scheduled_Delivery_Date: str,
                 Delivered_to_Client_Date: str,
                 Delivery_Recorded_Date: str,
                 Product_Group: str,
                 Sub_Classification: str,
                 Vendor: str,
                 Item_Description: str,
                 Molecule_Test_Type: str,
                 Brand: str,
                 Dosage: str,
                 Dosage_Form: str,
                 Unit_of_Measure_Per_Pack: int,
                 Line_Item_Quantity: int,
                 Line_Item_Value: float,
                 Pack_Price: float,
                 Unit_Price: float,
                 Manufacturing_Site: str,
                 First_Line_Designation: str,
                 Line_Item_Insurance_USD: float
                 ):
        try:
            self.country = country,
            self.Managed_By = Managed_By,
            self.Fulfill_Via = Fulfill_Via,
            self.Vendor_INCO_Term = Vendor_INCO_Term,
            self.Shipment_Mode = Shipment_Mode,
            self.Scheduled_Delivery_Date = Scheduled_Delivery_Date,
            self.Delivered_to_Client_Date = Delivered_to_Client_Date,
            self.Delivery_Recorded_Date = Delivery_Recorded_Date,
            self.Product_Group = Product_Group,
            self.Sub_Classification = Sub_Classification,
            self.Vendor = Vendor,
            self.Item_Description = Item_Description,
            self.Molecule_Test_Type = Molecule_Test_Type,
            self.Brand = Brand,
            self.Dosage = Dosage,
            self.Dosage_Form = Dosage_Form,
            self.Unit_of_Measure_Per_Pack = Unit_of_Measure_Per_Pack,
            self.Line_Item_Quantity = Line_Item_Quantity,
            self.Line_Item_Value = Line_Item_Value,
            self.Pack_Price = Pack_Price,
            self.Unit_Price = Unit_Price,
            self.Manufacturing_Site = Manufacturing_Site,
            self.First_Line_Designation = First_Line_Designation,
            self.Line_Item_Insurance_USD = Line_Item_Insurance_USD
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_shipment_input_data_frame(self):
        try:
            input_data = {
                "country": [self.country],
                "Managed_By": [self.Managed_By],
                "Fulfill_Via": [self.Fulfill_Via],
                "Vendor_INCO_Term": [self.Vendor_INCO_Term],
                "Shipment_Mode": [self.Shipment_Mode],
                "Scheduled_Delivery_Date": [self.Scheduled_Delivery_Date],
                "Delivered_to_Client_Date": [self.Delivered_to_Client_Date],
                "Delivery_Recorded_Date": [self.Delivery_Recorded_Date],
                "Product_Group": [self.Product_Group],
                "Sub_Classification": [self.Sub_Classification],
                "Vendor": [self.Vendor],
                "Item_Description": [self.Item_Description],
                "Molecule_Test_Type": [self.Molecule_Test_Type],
                "Brand": [self.Brand],
                "Dosage": [self.Dosage],
                "Dosage_Form": [self.Dosage_Form],
                "Unit_of_Measure_Per_Pack": [self.Unit_of_Measure_Per_Pack],
                "Line_Item_Quantity": [self.Line_Item_Quantity],
                "Line_Item_Value": [self.Line_Item_Value],
                "Pack_Price": [self.Pack_Price],
                "Unit_Price": [self.Unit_Price],
                "Manufacturing_Site": [self.Manufacturing_Site],
                "First_Line_Designation": [self.First_Line_Designation],
                "Line_Item_Insurance_USD": [self.Line_Item_Insurance_USD]}

            return pd.DataFrame(input_data)
        except Exception as e:
            raise ShipmentException(e, sys) from e


class ShipmentPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_latest_model_path(self, target: int):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[target]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def predict_t1(self, X):
        try:
            model_path = self.get_latest_model_path(0)
            model = load_object(file_path=model_path)
            shipment_pack_price = model.predict(X)
            return shipment_pack_price
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def predict_t2(self, X):
        try:
            model_path = self.get_latest_model_path(1)
            model = load_object(file_path=model_path)
            shipment_unit_price = model.predict(X)
            return shipment_unit_price
        except Exception as e:
            raise ShipmentException(e, sys) from e
