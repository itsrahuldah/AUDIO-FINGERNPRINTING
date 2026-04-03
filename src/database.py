from collections import defaultdict
import pickle

class SongDatabase:
    """
    In-memory inverted index database mapping hashes to (song_id, offset) pairs.
    Mimics a robust K-V store logic.
    """
    def __init__(self):
        # Inverted index: hash_value -> [(song_id, time_offset)]
        self.hashes = defaultdict(list)
        # Song lookup: song_id -> song_name/metadata
        self.songs = {}
        self.next_song_id = 1
        
    def add_song(self, song_name: str, hashes: list[tuple[int, int]]) -> int:
        song_id = self.next_song_id
        self.next_song_id += 1
        
        self.songs[song_id] = song_name
        
        for h_val, time_offset in hashes:
            self.hashes[h_val].append((song_id, time_offset))
            
        return song_id
        
    def query(self, h_val: int) -> list[tuple[int, int]]:
        return self.hashes.get(h_val, [])
        
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'hashes': dict(self.hashes), 'songs': self.songs}, f)
            
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.hashes = defaultdict(list, data['hashes'])
            self.songs = data['songs']
            self.next_song_id = max(self.songs.keys()) + 1 if self.songs else 1
