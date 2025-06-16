from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def dramasRecommendation(data, drama_title, filters, top_n=10):
    # Normalisasi title yang ada di database (jadi huruf kecil dan hilangkan spasi di depan dan belakang)
    data['normalized_title'] = data['Title'].str.lower().str.strip()

    # Normalisasi input dari user
    normalized_input = drama_title.lower().strip()

    # Cek apakah input title yang diberikan user ada di database, kalo gada return not found
    if normalized_input not in data['normalized_title'].values:
        return None, "Drama not found in database."

    # Ambil title aslinya yang belum di normalisasi
    actual_title = data.loc[data['normalized_title'] == normalized_input, 'Title'].values[0]
    
    # Sebelum masuk filter, drama yang di input user jangan dimasukin sebagai rekomendasi
    # Copy data yang sudah 'tidak ada title input dari user' untuk difilter 
    filtered_data = data[data['Title'] != actual_title].copy()

    if 'Genre' in filters:
        genre_string = data.loc[data['Title'] == actual_title, 'Genre'].values[0]
        # Ambil semua genre dari drama referensi
        genre_list = [g.strip().lower() for g in genre_string.split(',')]
        filtered_data = filtered_data[filtered_data['Genre'].str.lower().apply(
            lambda x: any(g in x for g in genre_list)
        )]
        print("After genre:", len(filtered_data))
    
    if 'Actors' in filters:
        actor_string = data.loc[data['Title'] == actual_title, 'Actors'].values[0]
        # Aktor yang main dalam sebuah drama ada lebih dari satu
        actor_list = [a.strip() for a in actor_string.split(',')]
        filtered_data = filtered_data[filtered_data['Actors'].apply(
            lambda x: any(a in x for a in actor_list) if pd.notna(x) else False
        )]
        print("After actor:", len(filtered_data))

    if 'Rating' in filters:
        rating = data.loc[data['Title'] == actual_title, 'Rating'].values[0]
        # Rating nya harus +0.5 atau -0.5 dari rating drama yang di input user
        filtered_data = filtered_data[
            (filtered_data['Rating'] >= rating - 0.5) &
            (filtered_data['Rating'] <= rating + 0.5)
        ]
        print("After rating:", len(filtered_data))
    
    if 'Description' in filters and not filtered_data.empty:
        reference_row = data[data['Title'] == actual_title].iloc[0:1]
        temp_data = pd.concat([reference_row, filtered_data], ignore_index=True)
        temp_data['describe'] = (
            "Title: " + temp_data["Title"] + "\n" +
            "Year of release: " + temp_data["Year of release"].astype(str) + "\n" +
            "Episodes: " + temp_data["Number of Episodes"].astype(str) + "\n" +
            "Rating: " + temp_data["Rating"].astype(str) + "\n" +
            "Description: " + temp_data["Description"] + "\n" +
            "Genre: " + temp_data["Genre"] + "\n" +
            "Tags: " + temp_data["Tags"]
        )
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(temp_data['describe']).toarray()
        similarity = cosine_similarity(vectors)

        distances = similarity[0]
        arr = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]

        recommendations = temp_data.iloc[[i[0] for i in arr]]
        print("After desc:", len(filtered_data))

        return recommendations[['Title', 'Genre', 'Rating']].sort_values(by='Rating', ascending=False), None

    else:
        if filtered_data.empty:
            return None, "No dramas matched your filters."
        else:
            return filtered_data[['Title', 'Genre', 'Rating']].head(top_n), None