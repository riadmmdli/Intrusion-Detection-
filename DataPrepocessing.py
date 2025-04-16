import pandas as pd
import os

base_path = "C:/Users/riadm/Desktop/Data Mining for Cyber Security Project Dataset"

folder_label_map = {
    '20200514_DNP3_Disable_Unsolicited_Messages_Attack': 'Disable_Unsolicited',
    '20200515_DNP3_Cold_Restart_Attack': 'Cold_Restart',
    '20200515_DNP3_Warm_Restart_Attack': 'Warm_Restart',
    '20200516_DNP3_Enumerate': 'Enumerate',
    '20200516_DNP3_Ιnfo': 'Info',  # note: check folder name for Greek "Ι"
    '20200518_DNP3_Initialize_Data_Attack': 'Initialize_Data',
    '20200518_DNP3_MITM_DoS': 'MITM_DoS',
    '20200518_DNP3_Replay_Attack': 'Replay',
    '20200519_DNP3_Stop_Application_Attack': 'Stop_Application',
}

merged_df = pd.DataFrame()

for folder, label in folder_label_map.items():
    csv_main = os.path.join(base_path, folder, "CSV Files")
    if not os.path.exists(csv_main):
        print(f"❌ CSV folder not found: {csv_main}")
        continue

    print(f"\n📁 Searching in: {csv_main}")

    # Go into each sub-subfolder like "All_Data", "Application_Layer_Only"
    for subfolder in os.listdir(csv_main):
        subfolder_path = os.path.join(csv_main, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"   📂 Checking subfolder: {subfolder}")
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        df["Class"] = label
                        merged_df = pd.concat([merged_df, df], ignore_index=True)
                        print(f"      ✅ Loaded: {file}, Rows: {df.shape[0]}")
                    except Exception as e:
                        print(f"      ❌ Failed: {file} | {e}")

print("\n🔧 Merging complete.")
print("🧪 Final shape of merged dataset:", merged_df.shape)
merged_df.to_csv("DNP3_Merged_Dataset.csv", index=False)
