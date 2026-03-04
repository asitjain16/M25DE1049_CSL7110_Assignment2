#!/usr/bin/env python3
"""
CSL7110 - Data Mining Assignment 2
Min-Hashing and Locality Sensitive Hashing
Author: Asit Jain
"""

import os
import time
import random
import hashlib
import itertools
import numpy as np
from collections import defaultdict

# ============================================================================
# Helper Functions
# ============================================================================

def load_document(filepath):
    """Load document from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def char_kgrams(text, k):
    """Generate character k-grams"""
    return set(text[i:i+k] for i in range(len(text) - k + 1))

def word_kgrams(text, k):
    """Generate word k-grams"""
    words = text.split()
    return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))

def jaccard(set_a, set_b):
    """Compute Jaccard similarity"""
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def gen_hash_funcs(t, m=10007):
    """Generate t hash functions h(x) = (a*x + b) % m"""
    funcs = []
    for _ in range(t):
        a = random.randint(1, m - 1)
        b = random.randint(0, m - 1)
        funcs.append((a, b, m))
    return funcs

def shingle_to_int(s):
    """Convert shingle to integer"""
    if isinstance(s, tuple):
        s = ' '.join(s)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

def minhash_sig(shingle_set, hash_funcs):
    """Compute min-hash signature"""
    t = len(hash_funcs)
    sig = [float('inf')] * t
    for s in shingle_set:
        v = shingle_to_int(s)
        for i, (a, b, m) in enumerate(hash_funcs):
            h = (a * v + b) % m
            if h < sig[i]:
                sig[i] = h
    return sig

def approx_jaccard(sig_a, sig_b):
    """Estimate Jaccard from signatures"""
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def load_movielens(path="ml-100k/u.data"):
    """Load MovieLens data"""
    user_movies = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            uid = int(parts[0])
            mid = int(parts[1])
            user_movies[uid].add(mid)
    return user_movies

def lsh_candidates(sigs, users, b, r):
    """Find candidate pairs using LSH banding"""
    candidates = set()
    for band in range(b):
        buckets = defaultdict(list)
        start = band * r
        end = start + r
        for u in users:
            portion = tuple(sigs[u][start:end])
            buckets[hash(portion)].append(u)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i+1, len(bucket)):
                        pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                        candidates.add(pair)
    return candidates

# ============================================================================
# QUESTION 1: K-Grams
# ============================================================================

def question1():
    print("=" * 80)
    print("QUESTION 1: K-Grams [20 POINTS]")
    print("=" * 80)
    
    # Load documents
    docs = {}
    for i in range(1, 5):
        docs[f"D{i}"] = load_document(f"minhash/D{i}.txt")
    
    kgrams = {}
    
    # Character 2-grams
    print("\n--- Character 2-grams ---")
    for name, text in docs.items():
        g = char_kgrams(text, 2)
        kgrams[(name, 'c2')] = g
        print(f"  {name}: {len(g)} unique 2-grams")
    
    # Character 3-grams
    print("\n--- Character 3-grams ---")
    for name, text in docs.items():
        g = char_kgrams(text, 3)
        kgrams[(name, 'c3')] = g
        print(f"  {name}: {len(g)} unique 3-grams")
    
    # Word 2-grams
    print("\n--- Word 2-grams ---")
    for name, text in docs.items():
        g = word_kgrams(text, 2)
        kgrams[(name, 'w2')] = g
        print(f"  {name}: {len(g)} unique 2-grams")
    
    # Jaccard similarities
    print("\n--- Jaccard Similarities ---")
    pairs = list(itertools.combinations(["D1","D2","D3","D4"], 2))
    
    for gtype, label in [('c2','Character 2-grams'), ('c3','Character 3-grams'), ('w2','Word 2-grams')]:
        print(f"\n{label}:")
        for d1, d2 in pairs:
            sim = jaccard(kgrams[(d1, gtype)], kgrams[(d2, gtype)])
            print(f"  J({d1},{d2}) = {sim:.6f}")
    
    return kgrams

# ============================================================================
# QUESTION 2: Min-Hashing
# ============================================================================

def question2():
    print("\n" + "=" * 80)
    print("QUESTION 2: Min-Hashing [20 POINTS]")
    print("=" * 80)
    
    # Load D1 and D2 with 3-grams
    d1 = char_kgrams(load_document("minhash/D1.txt"), 3)
    d2 = char_kgrams(load_document("minhash/D2.txt"), 3)
    exact = jaccard(d1, d2)
    
    print(f"\nExact Jaccard(D1, D2) with 3-grams: {exact:.6f}")
    
    # Part A: Different values of t
    print("\n--- Part A: Approximate Jaccard for different t ---")
    print(f"{'t':>6}  {'Approx':>10}  {'Exact':>10}  {'Error':>10}  {'Time(s)':>10}")
    
    for t in [20, 60, 150, 300, 600]:
        random.seed(42)
        start = time.time()
        hf = gen_hash_funcs(t)
        s1 = minhash_sig(d1, hf)
        s2 = minhash_sig(d2, hf)
        aj = approx_jaccard(s1, s2)
        elapsed = time.time() - start
        print(f"{t:>6}  {aj:>10.6f}  {exact:>10.6f}  {abs(aj-exact):>10.6f}  {elapsed:>10.4f}")
    
    # Part B: Finding good t value
    print("\n--- Part B: Finding optimal t value ---")
    print(f"{'t':>6}  {'Avg Error':>12}  {'Avg Time':>12}")
    
    for t in [20, 40, 60, 80, 100, 150, 200, 300, 400, 600]:
        errors, times = [], []
        for run in range(5):
            random.seed(run * 100 + t)
            hf = gen_hash_funcs(t)
            start = time.time()
            s1 = minhash_sig(d1, hf)
            s2 = minhash_sig(d2, hf)
            aj = approx_jaccard(s1, s2)
            times.append(time.time() - start)
            errors.append(abs(aj - exact))
        print(f"{t:>6}  {np.mean(errors):>12.6f}  {np.mean(times):>12.4f}")
    
    print("\nRecommendation: t=150-200 provides good accuracy/speed tradeoff")

# ============================================================================
# QUESTION 3: LSH
# ============================================================================

def question3():
    print("\n" + "=" * 80)
    print("QUESTION 3: LSH [20 POINTS]")
    print("=" * 80)
    
    t = 160
    tau = 0.7
    
    def s_curve(s, b, r):
        """LSH probability function"""
        return 1.0 - (1.0 - s**r)**b
    
    # Part A: Find best b, r
    print(f"\n--- Part A: Finding optimal b and r for t={t}, tau={tau} ---")
    print(f"{'b':>4}  {'r':>4}  {'Threshold':>12}  {'|tau-thresh|':>14}")
    
    best_b, best_r, best_score = None, None, float('inf')
    
    for b in range(1, t+1):
        if t % b != 0:
            continue
        r = t // b
        threshold = (1.0/b) ** (1.0/r)
        score = abs(threshold - tau)
        marker = " <-- BEST" if score < best_score else ""
        print(f"{b:>4}  {r:>4}  {threshold:>12.6f}  {score:>14.6f}{marker}")
        if score < best_score:
            best_score = score
            best_b, best_r = b, r
    
    print(f"\nOptimal: b={best_b}, r={best_r}")
    print(f"Threshold: (1/{best_b})^(1/{best_r}) = {(1.0/best_b)**(1.0/best_r):.6f}")
    
    # Part B: Probability for each pair
    print(f"\n--- Part B: Candidate probabilities (b={best_b}, r={best_r}) ---")
    
    docs = {}
    for i in range(1, 5):
        docs[f"D{i}"] = char_kgrams(load_document(f"minhash/D{i}.txt"), 3)
    
    pairs = list(itertools.combinations(["D1","D2","D3","D4"], 2))
    print(f"{'Pair':>8}  {'Jaccard':>10}  {'P(candidate)':>14}  {'Above tau?':>12}")
    
    for d1, d2 in pairs:
        j = jaccard(docs[d1], docs[d2])
        prob = s_curve(j, best_b, best_r)
        above = 'Yes' if j > tau else 'No'
        print(f"{d1}-{d2:>5}  {j:>10.6f}  {prob:>14.6f}  {above:>12}")

# ============================================================================
# QUESTION 4: Min-Hashing on MovieLens
# ============================================================================

def question4():
    print("\n" + "=" * 80)
    print("QUESTION 4: Min-Hashing on MovieLens [20 POINTS]")
    print("=" * 80)
    
    data_path = "ml-100k/u.data"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at '{data_path}'")
        print("Download from: http://files.grouplens.org/datasets/movielens/ml-100k.zip")
        return
    
    user_movies = load_movielens(data_path)
    users = sorted(user_movies.keys())
    n = len(users)
    print(f"\nLoaded {n} users")
    
    # Compute exact Jaccard using inverted index (much faster)
    print("Computing exact Jaccard similarities...")
    start = time.time()
    
    movie_users = defaultdict(set)
    for u in users:
        for m in user_movies[u]:
            movie_users[m].add(u)
    
    pair_inter = defaultdict(int)
    for movie, uset in movie_users.items():
        ulist = sorted(uset)
        for i in range(len(ulist)):
            for j in range(i+1, len(ulist)):
                pair_inter[(ulist[i], ulist[j])] += 1
    
    exact_above = set()
    exact_sims = {}
    for (u1, u2), inter in pair_inter.items():
        union = len(user_movies[u1]) + len(user_movies[u2]) - inter
        sim = inter / union if union > 0 else 0.0
        if sim >= 0.5:
            exact_above.add((u1, u2))
            exact_sims[(u1, u2)] = sim
    
    print(f"Done in {time.time()-start:.2f}s")
    print(f"Pairs with Jaccard >= 0.5: {len(exact_above)}")
    
    # Show first 10 pairs
    if exact_above:
        print("\nFirst 10 pairs:")
        for u1, u2 in sorted(exact_above)[:10]:
            print(f"  ({u1:>3}, {u2:>3}): {exact_sims[(u1,u2)]:.6f}")
    
    # Min-hash approximation
    for t in [50, 100, 200]:
        print(f"\n--- t = {t} hash functions ---")
        fps, fns = [], []
        
        for run in range(5):
            random.seed(run * 1000 + t)
            hf = gen_hash_funcs(t)
            
            # Build signatures
            sigs = {}
            for u in users:
                sig = [float('inf')] * t
                for movie in user_movies[u]:
                    for idx, (a, b, m) in enumerate(hf):
                        h = (a * movie + b) % m
                        if h < sig[idx]:
                            sig[idx] = h
                sigs[u] = sig
            
            # Find pairs with approx jaccard >= 0.5
            approx_above = set()
            for i in range(n):
                for j in range(i+1, n):
                    u1, u2 = users[i], users[j]
                    matches = sum(1 for a, b in zip(sigs[u1], sigs[u2]) if a == b)
                    if matches / t >= 0.5:
                        approx_above.add((u1, u2))
            
            fp = len(approx_above - exact_above)
            fn = len(exact_above - approx_above)
            fps.append(fp)
            fns.append(fn)
            print(f"  Run {run+1}: FP={fp}, FN={fn}")
        
        print(f"  Average: FP={np.mean(fps):.2f}, FN={np.mean(fns):.2f}")

# ============================================================================
# QUESTION 5: LSH on MovieLens
# ============================================================================

def question5():
    print("\n" + "=" * 80)
    print("QUESTION 5: LSH on MovieLens [20 POINTS]")
    print("=" * 80)
    
    data_path = "ml-100k/u.data"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at '{data_path}'")
        print("Download from: http://files.grouplens.org/datasets/movielens/ml-100k.zip")
        return
    
    user_movies = load_movielens(data_path)
    users = sorted(user_movies.keys())
    
    # Compute exact Jaccard
    print("Computing exact Jaccard similarities...")
    movie_users = defaultdict(set)
    for u in users:
        for m in user_movies[u]:
            movie_users[m].add(u)
    
    pair_inter = defaultdict(int)
    for movie, uset in movie_users.items():
        ulist = sorted(uset)
        for i in range(len(ulist)):
            for j in range(i+1, len(ulist)):
                pair_inter[(ulist[i], ulist[j])] += 1
    
    exact_sim = {}
    for (u1, u2), inter in pair_inter.items():
        union = len(user_movies[u1]) + len(user_movies[u2]) - inter
        exact_sim[(u1, u2)] = inter / union if union > 0 else 0.0
    
    # Test configurations
    configs = [
        (100, 20, 5),   # t=100, b=20, r=5
        (100, 10, 10),  # t=100, b=10, r=10
        (100, 5, 20),   # t=100, b=5, r=20
        (200, 20, 10),  # t=200, b=20, r=10
    ]
    
    for tau in [0.6, 0.8]:
        print(f"\n{'='*80}")
        print(f"Threshold tau = {tau}")
        print(f"{'='*80}")
        
        exact_above = {(u1,u2) for (u1,u2), s in exact_sim.items() if s >= tau}
        print(f"Exact pairs with Jaccard >= {tau}: {len(exact_above)}")
        
        for t, b, r in configs:
            print(f"\n--- Configuration: t={t}, b={b}, r={r} ---")
            fps, fns = [], []
            
            for run in range(5):
                random.seed(run * 2000 + t + b)
                hf = gen_hash_funcs(t)
                
                # Build signatures
                sigs = {}
                for u in users:
                    sig = [float('inf')] * t
                    for movie in user_movies[u]:
                        for idx, (a, bc, m) in enumerate(hf):
                            h = (a * movie + bc) % m
                            if h < sig[idx]:
                                sig[idx] = h
                    sigs[u] = sig
                
                # LSH candidates
                cands = lsh_candidates(sigs, users, b, r)
                
                # Filter by threshold
                approx_above = set()
                for u1, u2 in cands:
                    matches = sum(1 for a, b in zip(sigs[u1], sigs[u2]) if a == b)
                    if matches / t >= tau:
                        approx_above.add((u1, u2))
                
                fp = len(approx_above - exact_above)
                fn = len(exact_above - approx_above)
                fps.append(fp)
                fns.append(fn)
            
            print(f"  Average: FP={np.mean(fps):.2f}, FN={np.mean(fns):.2f}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CSL7110 - DATA MINING ASSIGNMENT 2")
    print("Min-Hashing and Locality Sensitive Hashing")
    print("="*80 + "\n")
    
    question1()
    question2()
    question3()
    question4()
    question5()
    
    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETED")
    print("="*80)
