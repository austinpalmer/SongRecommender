import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Ready only necessary columns for recommendation
df = pd.read_csv("data/topSongs.csv", usecols=["title", "artist"], low_memory=False)

# Clean the dataframe
df = df.drop_duplicates(subset="title")  # remove duplicates
df = df.dropna(axis=0)                     # drop null values

# Replace spaces in artist name to read full value
df["artist"] = df["artist"].str.replace(" ", "")
# Combine all columns and assign as new column
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

# Initialize CountVectorizer and create a sparse matrix
vectorizer = CountVectorizer(binary=True)
vectorized = vectorizer.fit_transform(df["data"])
sparse_matrix = csr_matrix(vectorized)
# Gather similarities in songs using cosine similarity
similarities = cosine_similarity(sparse_matrix)
# Assign the new dataframe with 'similarities' values
df_temp = pd.DataFrame(similarities, columns=df["title"], index=df["title"]).reset_index()

true = True
while true:
    print("The Top 10 Song Recommendation System")
    print("-------------------------------------")
    print("This will generate 10 songs that are similar to the inputted song")

    while True:
        input_song = input("Please enter the name of the song: ")

        if input_song in df_temp.columns:
            recommendation = df_temp.nlargest(11, input_song)["title"]
            break
        else:
            print("Sorry, that song is not in our database. Please try again")

    print("You should check out these songs: \n")
    for song in recommendation.values[1:]:
        print(song)
    print("\n")

    # Ask the user for the next command
    while True:
        next_command = input("Do you want to generate again for the next song? [yes, no] ")

        if next_command == "yes":
            break
        elif next_command == "no":
            # End program
            true = False
            break
        else:
            print("Please type 'yes' or 'no'")