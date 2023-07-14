% Init
coord(X) :- X = 0.0; X = 1.0; X = 2.0; X = 3.0.
coord(Y) :- Y = 0.0; Y = 1.0; Y = 2.0.
admissible_cells(X,Y) :- coord(X), coord(Y).


% Perception
nn(perc_net_version_1_NeSy_0,[Img,X,Y],Z,[human,nothing]) :: cell(Img,X,Y,Z).


% Prediction
attention_boundary(X,Y) :- X==1.0, Y==0.0.


% Prediction
enlarged_boundary(X,Y) :- X==1.0, Y==1.0.
enlarged_boundary(X,Y) :- X==2,0, Y==0.0.
enlarged_boundary(X,Y) :- X==2.0, Y==1.0.


% Prediction ---> ... pixels
out_of_view(X,Y) :- X==0.0; X==3.0; Y==2.0.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- admissible_cells(X,Y), attention_boundary(X,Y), cell(Img,X,Y,human).
action(Img,2) :- admissible_cells(X,Y), enlarged_boundary(X,Y), cell(Img,X,Y,human), \+ action(Img,1).
action(Img,0) :- admissible_cells(X,Y), out_of_view(X,Y), \+ action(Img,2), \+ action(Img,1).
