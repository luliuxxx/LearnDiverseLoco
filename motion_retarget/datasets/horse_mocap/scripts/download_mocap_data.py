import requests

def download_file(url, destination):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully to {destination}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.InvalidURL as e:
        print(f"An error occurred: {e}")

# Example usage:
# SEE: http://horse.cs.uni-bonn.de/dataset1.html

possible_id = [1,2,3]
for h in range(1,15): # possible options: 1,2,3,4,5,6,7,8,9,10,11,12,13,14
    for day in possible_id:
        # measurement_day = i # possible options: 1,2,3
        for id in possible_id:# walk # possible options: 1,2,3
            for type in ["walk","trot"]:
                filename = f"Horse{h}_M{day}_{type}{id}_kinematics.csv"
                url = f"http://horse.cs.uni-bonn.de/dataset/{filename}"  # Replace with the actual URL of the file you want to download
                destination = f"datasets/horse_mocap/raw_mocap_data/{filename}"
                download_file(url, destination)
