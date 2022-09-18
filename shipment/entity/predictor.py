import os
import sys
from shipment.constant import ROOT_DIR
from shipment.exception import ShipmentException
from shipment.utils.util import load_object

import pandas as pd


def transform(X: pd.DataFrame):
    try:
        df: pd.DataFrame = X.copy()

        columns_remove = ['ID', 'PQ First Sent to Client Date', 'PO Sent to Vendor Date', 'Weight (Kilograms)',
                          'Freight Cost (USD)', 'PQ #', 'PO / SO #', 'ASN/DN #']
        for i in columns_remove:
            if i in df:
                df.drop(i, axis=1, inplace=True)

        for column in ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']:
            if column in df:
                df[column] = pd.to_datetime(df[column])
                df[column + ' Year'] = df[column].apply(lambda x: x.year)
                df[column + ' Month'] = df[column].apply(lambda x: x.month)
                df[column + ' Day'] = df[column].apply(lambda x: x.day)
                df = df.drop(column, axis=1)

        binary_columns = ['Fulfill Via', 'First Line Designation']
        for i in binary_columns:
            if i in df:
                df['Fulfill Via'] = df['Fulfill Via'].replace({'Direct Drop': 0, 'From RDC': 1})
                df['First Line Designation'] = df['First Line Designation'].replace({'No': 0, 'Yes': 1})

        # Fill missing values
        df['Shipment Mode'] = df['Shipment Mode'].fillna(df['Shipment Mode'].mode()[0])
        df['Dosage'] = df['Dosage'].fillna(df['Dosage'].mode()[0])
        df['Line Item Insurance (USD)'] = df['Line Item Insurance (USD)'].fillna(
            df['Line Item Insurance (USD)'].mean())

        object_type = ['Country', 'Managed By', 'Vendor INCO Term',
                       'Shipment Mode', 'Product Group', 'Sub Classification', 'Vendor',
                       'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage',
                       'Dosage Form', 'Manufacturing Site', 'Project Code']
        # One-hot encoding
        for column in object_type:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)

        df = df.iloc[-1].to_frame()
        return df.T
    except Exception as e:
        raise ShipmentException(e, sys)


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
                "country": [self.country],
                "Managed By": [self.Managed_By],
                "Fulfill Via": [self.Fulfill_Via],
                "Vendor INCO Term": [self.Vendor_INCO_Term],
                "Shipment Mode": [self.Shipment_Mode],
                "Scheduled Delivery Date": [self.Scheduled_Delivery_Date],
                "Delivered to Client Date": [self.Delivered_to_Client_Date],
                "Delivery Recorded Date": [self.Delivery_Recorded_Date],
                "Product Group": [self.Product_Group],
                "Sub Classification": [self.Sub_Classification],
                "Vendor": [self.Vendor],
                "Item Description": [self.Item_Description],
                "Molecule/Test Type": [self.Molecule_Test_Type],
                "Brand": [self.Brand],
                "Dosage": [self.Dosage],
                "Dosage Form": [self.Dosage_Form],
                "Unit of Measure (Per Pack)": [self.Unit_of_Measure_Per_Pack],
                "Line Item Quantity": [self.Line_Item_Quantity],
                "Line Item Value": [self.Line_Item_Value],
                "Pack Price": [self.Pack_Price],
                "Manufacturing Site": [self.Manufacturing_Site],
                "First Line Designation": [self.First_Line_Designation],
                "Line Item Insurance (USD)": [self.Line_Item_Insurance_USD]}

            train = os.path.join(ROOT_DIR, "SCMS_Delivery_History_Dataset.csv")
            train_df = pd.read_csv(train).drop("Unit Price", axis=1)
            input_df = pd.read_csv(input_data)
            df = pd.concat(objs=[train_df, input_df], axis=0)
            tran_df = transform(df)
            return tran_df
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

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path(0)
            model = load_object(file_path=model_path)
            shipment_pack_price = model.predict(X)
            return shipment_pack_price
        except Exception as e:
            raise ShipmentException(e, sys) from e
