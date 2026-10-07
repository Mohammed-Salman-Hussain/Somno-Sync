import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file inside the dataset
file_path = "bedtime_screentime_sleep_debt.csv"

# Load the dataset using kagglehub
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "samartalwar/sleep-debt-and-screen-time-late-night-phone-habits",
    file_path,
)

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 records:")
print(df.head())
