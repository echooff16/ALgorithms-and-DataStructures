# Algorithm and Datastructure Project

## 1 Logistics & Resource Optimisation System (Python)  

You are organising a study trip for students to an industry partner in another state. The
industry partner would only support the trip if there are sufficient students joining. You are
providing transport for the students in the form of buses, but the students would need to make
their own way to the various pickup points in the city.

As a business-focused person, you know of all the different possible combinations to assign
buses to pickup points in the city:

- There are L number of locations in the city. The location IDs are integers in the range
    of 0 , 1 , ..., L− 1. There can only be up to 18 designated pickup points in the city.
- There are R bidirectional roads connecting locations in the city. There can only be a
    single road connecting any 2 locations. However, there is no guarantee that a location is
    reachable from another location through one or more roads.
- There are S number of students living in the city. The student IDs are integers in the
    range of 0 , 1 , ..., S− 1 and each student is located at one of the locations. There can be
    more than a single student at any given location.
- There are B number of buses to be used to fetch the students, each to be located at
    one of the designated pickup points in the city. The bus IDs are integers in the range of
    0 , 1 , ..., B− 1. The pickup point for each bus is already fixed; and there can be more than
    one bus assigned to a single pickup point. It is possible that not all 18 pickup points in
    the city will be used even if there are more than 18 buses.
- The students are only willing to travel up to a distance of D from their location to a
    pickup point to board a bus. Any pickup point is fine for them as long as it is within
    D distance; but if there are no pickup points within D distance, then they would not
    participate in the trip. D is a positive integer.
- Each bus has a minimum and a maximum capacity, inclusive, that need to be respected.
    Both the minimum and maximum capacities are positive integers.
- The goal is to check if the current bus allocations will be able to fetch exactly T students
    for the trip.T is a positive integer.
- Since you have already paid for all the buses, you require all of the buses to be used.


Write a functionassign(L, roads, students, buses, D, T)tosolve this using the knowl-
edge that you have gained from one or more FIT2004 topics. The inputs of function
assign are as follows:

- L is a positive integer denoting the number of locations in the city.
- roadsis a list of roads where each road is represented as a tuple(u, v, w)where:
    - u is the starting location of the road,
    - v is the ending location of the road,
    - w is the length of the road.
- students is a list of numbers denoting the location of each student in the city.
- buses is a list of buses where each bus is represented as a tuple(e, f, g)where:
    - e is the pickup location of the bus,
    - f is the minimum capacity of the bus,
    - g is the maximum capacity of the bus.
- D is a positive integer denoting the maximum distance students are willing to travel to
    pickup points.
- T is a positive integer denoting the exact number of students that need to travel in a
    valid solution.

Your task is to write assign that solves this allocation problem:

- In case no allocation satisfying all constraints exists, your algorithm should return None
    (i.e., Python NoneType).
- Otherwise, it should return a list of integers allocation of lengthS containing one
    possible allocation of the students to the buses that satisfies all constraints. Fori ∈
    0 , 1 ,... , S− 1 , allocation[i]denotes the bus number to which student iwould be
    allocated if it is a non-negative integer, and denotes that the student is not travelling if it is
    equal to -1.

### 1.1 Complexity Requirements

The function assign(L, roads, students, buses, D, T) should have a worst-case time
complexity of O (S·T+L+R·logL)and a worst-case auxiliary space complexity of O(S+L+R).



### 1.2 Example(s)

Consider the following simple examples in which the function returns one of the possible valid
assignments for the given input if possible. Else, it returns None.

```
Figure 1: Visualization of the Map.
```
```
# Refer to the visualization for the map.
# There are 16 locations in total and 15 roads.
L = 16
roads = [(0,1,3), (0,2,5), (0,3,10), (1,4,1), (2,5,2), (5,6,3),
(2,7,4), (0,8,1), (0,9,1), (0,10,1), (0,11,1), (6,12,2), (6,13,4),
(6,14,3), (7,15,1)]
```

# The location of each student in the city.
students = [4, 10, 8, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 5, 7, 7,
7, 7, 7, 15, 15, 7, 4, 8, 9]

# The buses as tuple (pickup location, minimum capacity, maximum capacity).
# Example: Bus-1 will pickup the students at location-6 and should transport
# a minimum of 5 students and a maximum of 10 students inclusive. Bus-3 is
# observed to have the same arrangements as well.
buses = [(0, 3, 5), (6, 5, 10), (15, 5, 10), (6, 5, 10)]

# The maximum distance a student is willing to travel to pickup point.
D = 5

# The exact number of students required.
T = 22

# Calling the function
>>> assign(L, roads, students, buses, D, T)
[0, 0, -1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, -1, -1, 2, 2, 2, 2, 2, 2, 2, 0,
0, 0]
# The non-negative integers are the bus IDs that will be boarded by the
# students if they are going on the trip. It is -1 otherwise.

# The returned assignment is not unique. An alternative assignment can also
# be [0, -1, 0, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, -1, -1, 2, 2, 2, 2, 2, 2, 2,
# 0, 0, 0]. There are other alternative assignments.

students = [5, 8, 3, 7, 7, 15, 15, 8, 15, 7, 6, 15]
buses = [(0, 3, 5), (15, 5, 6)]
D = 5
T = 7

# Calling the function for the same city (locations and roads).
>>> assign(L, roads, students, buses, D, T)
None
# Return None when there is no valid assignments.


## 2 Music Pattern Analytics Tool (Python)

You are working for a music streaming company that aims to understand what patterns in
melodies capture listeners’ attention. Every note in a song tells part of a story, and sequences
of notes often reveal recurring motifs that make a song catchy or memorable. To uncover these
hidden patterns, you decide to analyse sequences of musical notes from different songs.
Here are some example note sequences from several songs:

- Song 1: C→E→G→E→C
- Song 2: G→C→E→G→B
- Song 3: C→D→E→G→C

While exploring these sequences, you notice some intriguing similarities. For instance, both
Song 1 and Song 2 contain the note sequence C→E→G, suggesting that this motif may
serve as a common building block in popular melodies.

Inspired by this observation, you propose to your manager that the music recommendation
system could highlight songs sharing such motifs, helping listeners discover music with familiar
or appealing patterns. Your manager is enthusiastic and wants to identify more of these hidden
patterns in the data. Your task is therefore to analyse the note sequences and find subsequences
of notes that repeat across multiple songs.

In the company’s database, each record corresponds to a sequence of notes representing a song.
Each sequence may contain many notes, making direct analysis computationally intensive. To
simplify the task, you focus on a restricted range of commonly used notes. You may assume
that all notes have been mapped to the 26 lowercase English letters, from a to z.

After preprocessing, each song’s note sequence can be represented as:

```
S=⟨n 1 , n 2 ,... , nm⟩,
```
where m is the length of the sequence and n i is the i-th note. A pattern of length k is defined
as a contiguous subsequence
P=⟨np, 1 , np, 2 ,... , np,k⟩,

that appears in one or more songs. The objective is to identify patterns of various lengths that
occur in the largest number of distinct songs. Detecting such frequently recurring motifs can
reveal common musical structures and support data-driven recommendations.

Note that although a pattern may occur multiple times within the same song, it is counted
only once per song. Furthermore, patterns that appear with a key change are also considered
occurrences. A key change corresponds to a transposition of the pattern, where every note is
shifted by a fixed interval. That is, a patternP+i=⟨np, 1 +i, np, 2 +i,... , np,k+i⟩is considered
as an occurrence ofP =⟨np, 1 , np, 2 ,... , np,k⟩, whereiis an integer. For example, the pattern
“acd” with a key change of +2 is “cef”. Finally, the transposition is not modular. For example,
the pattern “za” does not match the pattern “ab”.

You should implement a class Analyser which provides the following functionality:


- __init__(self, sequences): The constructor takes the input sequences, which is the
    list of strings representing note sequences. The constructor preprocesses all note sequences
    for later analysis. Its time and space complexity should be bounded by O(N M^2 ), where:
       - N is the number of note sequences insequences, and
       - M is the length of the longest note sequence.

```
After preprocessing, the total space used by the Analyser object should be bounded by
O(N M).
```
- getFrequentPattern(self, K): Returns a list of characters representing the most frequent-
    quent pattern of length K across all note sequences, where 2 ≤ K ≤M. If multiple
    patterns have the same highest frequency, return one of them. This method should have
    worst-case time complexity O (K).

Example Output:

```
>> demo_songs = ["cegec", "gdfhd", "cdfhd"]
>> analyser = Analyser(demo_songs)
```
```
# The most frequent pattern of length 2.
# Other return values are possible, e.g. ["c", "e"]
>> print("K=2 =>", analyser.getFrequentPattern(2))
K=2 => ["d", "f"]
```
```
# The most frequent pattern of length 3.
# Other return values are possible, e.g. ["c", "e", "g"]
>> print("K=3 =>", analyser.getFrequentPattern(3))
K=3 => ["d", "f", "h"]
```
```
# The most frequent pattern of length 4.
# Other return values are not possible.
>> print("K=4 =>", analyser.getFrequentPattern(4))
K=4 => ["d", "f", "h", "d"]
```

