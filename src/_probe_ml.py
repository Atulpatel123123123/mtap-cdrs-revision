from src.movielens import load_ml1m, build_genre_pair
import itertools

r, m = load_ml1m()
m2 = m.copy()
m2["Primary"] = m2["Genres"].str.split("|").str[0]
print(m2["Primary"].value_counts().head(15))
genres = list(m2["Primary"].value_counts().head(8).index)
for a, b in itertools.combinations(genres, 2):
    S, T, meta = build_genre_pair(r, m, a, b, min_ratings=20)
    if meta["n_users"] >= 200:
        print(
            f"OK {a}->{b} users={meta['n_users']} dens_t={meta['density_t']:.4f} "
            f"src={meta['source']} tgt={meta['target']}"
        )
