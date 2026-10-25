# https://docs.pasqal.com/pulser/tutorials/qpu/
import os

from dotenv import load_dotenv
from pasqal_cloud import PasqalCloudConnection

# Load the .env file
load_dotenv()

USERNAME = os.getenv("USERNAME")
PROJECT_ID = os.getenv("PROJECT_ID")
PASSWORD = os.getenv("PASSWORD")


connection = PasqalCloudConnection(
    username=USERNAME,  # Your username or email address for the Pasqal Cloud Platform
    project_id=PROJECT_ID,  # The ID of the project associated to your account
    password=PASSWORD,  # The password for your Pasqal Cloud Platform account
)
print(connection.fetch_available_devices())
device = connection.fetch_available_devices()["FRESNEL_CAN1"]

print(device.specs)
