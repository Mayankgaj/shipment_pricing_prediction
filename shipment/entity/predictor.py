import os, sys
import numpy as np
from shipment.entity.prediction_data import call_empty
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
            self.Manufacturing_Site = Manufacturing_Site,
            self.First_Line_Designation = First_Line_Designation,
            self.Line_Item_Insurance_USD = Line_Item_Insurance_USD
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def get_shipment_input_data_frame(self):
        try:
            input_data = {
                "Country": list(self.country),
                "Managed By": list(self.Managed_By),
                "Fulfill Via": list(self.Fulfill_Via),
                "Vendor INCO Term": list(self.Vendor_INCO_Term),
                "Shipment Mode": list(self.Shipment_Mode),
                "Scheduled Delivery Date": list(self.Scheduled_Delivery_Date),
                "Delivered to Client Date": list(self.Delivered_to_Client_Date),
                "Delivery Recorded Date": list(self.Delivery_Recorded_Date),
                "Product Group": list(self.Product_Group),
                "Sub Classification": list(self.Sub_Classification),
                "Vendor": list(self.Vendor),
                "Item Description": list(self.Item_Description),
                "Molecule/Test Type": list(self.Molecule_Test_Type),
                "Brand": list(self.Brand),
                "Dosage": list(self.Dosage),
                "Dosage Form": list(self.Dosage_Form),
                "Unit of Measure (Per Pack)": list(self.Unit_of_Measure_Per_Pack),
                "Line Item Quantity": list(self.Line_Item_Quantity),
                "Line Item Value": list(self.Line_Item_Value),
                "Pack Price": list(self.Pack_Price),
                "Manufacturing Site": list(self.Manufacturing_Site),
                "First Line Designation": list(self.First_Line_Designation),
                "Line Item Insurance (USD)": float(self.Line_Item_Insurance_USD)}

            input_df = pd.DataFrame.from_dict(input_data)
            input_df.to_csv("input_raw.csv")
            date_col = ["Scheduled Delivery Date", "Delivered to Client Date",
                        "Delivery Recorded Date"]
            empty_data = call_empty()
            input_df['Fulfill Via'] = input_df['Fulfill Via'].replace({'Direct Drop': 0, 'From RDC': 1})
            input_df['First Line Designation'] = input_df['First Line Designation'].replace({'No': 0, 'Yes': 1})
            for column in date_col:
                my_date = pd.to_datetime(input_df[column][0])
                empty_data[column + " Year"] = my_date.year
                empty_data[column + " Month"] = my_date.month
                empty_data[column + " Day"] = my_date.day
                input_df.drop(column, axis=1, inplace=True)

            col = ['Country', 'Managed By', 'Fulfill Via', 'Vendor INCO Term', 'Shipment Mode', 'Product Group',
                   'Sub Classification', 'Vendor', 'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage',
                   'Dosage Form', 'Unit of Measure (Per Pack)', 'Line Item Quantity', 'Line Item Value', 'Pack Price',
                   'Manufacturing Site', 'First Line Designation', 'Line Item Insurance (USD)']
            for i in col:
                if type(input_df[i][0]) == (np.int64 or np.float64):
                    empty_data[i] = input_df[i][0]
                elif type(input_df[i][0]) == str:
                    empty_data[i + "_" + str(input_df[i][0])] = 1

            empty_data["Pack Price"] = input_df["Pack Price"][0]
            tran_input = pd.DataFrame.from_dict(empty_data, orient='index').T
            tran_input["t"] = 0
            tran_input["s"] = 0
            return np.array(tran_input)
        except Exception as e:
            raise ShipmentException(e, sys) from e

    def data(self):
        input_data = {
            "Country": list(self.country),
            "Managed By": list(self.Managed_By),
            "Fulfill Via": list(self.Fulfill_Via),
            "Vendor INCO Term": list(self.Vendor_INCO_Term),
            "Shipment Mode": list(self.Shipment_Mode),
            "Scheduled Delivery Date": list(self.Scheduled_Delivery_Date),
            "Delivered to Client Date": list(self.Delivered_to_Client_Date),
            "Delivery Recorded Date": list(self.Delivery_Recorded_Date),
            "Product Group": list(self.Product_Group),
            "Sub Classification": list(self.Sub_Classification),
            "Vendor": list(self.Vendor),
            "Item Description": list(self.Item_Description),
            "Molecule/Test Type": list(self.Molecule_Test_Type),
            "Brand": list(self.Brand),
            "Dosage": list(self.Dosage),
            "Dosage Form": list(self.Dosage_Form),
            "Unit of Measure (Per Pack)": list(self.Unit_of_Measure_Per_Pack),
            "Line Item Quantity": list(self.Line_Item_Quantity),
            "Line Item Value": list(self.Line_Item_Value),
            "Pack Price": list(self.Pack_Price),
            "Manufacturing Site": list(self.Manufacturing_Site),
            "First Line Designation": list(self.First_Line_Designation),
            "Line Item Insurance (USD)": float(self.Line_Item_Insurance_USD)}

        return pd.DataFrame(input_data)


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

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path(0)
            model = load_object(file_path=model_path)
            shipment_pack_price = model.predict(X)
            return shipment_pack_price
        except Exception as e:
            raise ShipmentException(e, sys) from e
