An implementation of the PageRank algorithm to rank college football teams based on margin of victory and home/away advantage.

Teams are related to each other (imagine each team being a node in a graph, and their relation being the edges) according to the margin of victory between the teams (adjusted for home/away advantage). A "smoothing" function is applied to the margin of victory to give diminishing reward as margin of victory increases (i.e. it is concave down for MOV > 0; it is similarly concave up for MOV < 0). Some liberty can be taken for this function of the margin of victory.
