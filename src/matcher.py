from collections import defaultdict
from src.database import SongDatabase

def match_hashes(query_hashes: list[tuple[int, int]], db: SongDatabase) -> tuple[int, int, int]:
    """
    Matches query hashes against the database using offset histogram voting.
    
    Robust to noise: Random hash collisions will not share consistent time-deltas.
    True matches align on `db_offset - query_offset = constant`.
    
    Args:
        query_hashes: List of (hash_val, query_offset).
        db: The initialized SongDatabase.
        
    Returns:
        best_song_id: Identified song ID.
        best_offset: Time offset (frames).
        confidence: Max vote count in the histogram.
    """
    # histogram[song_id][dt] = count
    song_scores = defaultdict(lambda: defaultdict(int))
    
    for h_val, query_offset in query_hashes:
        matches = db.query(h_val)
        for (song_id, db_offset) in matches:
            dt = db_offset - query_offset
            song_scores[song_id][dt] += 1
            
    if not song_scores:
        return None, None, 0
        
    best_song_id = None
    best_offset = None
    max_votes = 0
    
    for song_id, dt_hist in song_scores.items():
        best_dt = max(dt_hist.keys(), key=lambda t: dt_hist[t])
        votes = dt_hist[best_dt]
        
        if votes > max_votes:
            max_votes = votes
            best_song_id = song_id
            best_offset = best_dt
            
    return best_song_id, best_offset, max_votes
