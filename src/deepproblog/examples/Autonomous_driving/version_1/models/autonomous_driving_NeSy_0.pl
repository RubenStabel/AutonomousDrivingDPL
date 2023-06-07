% Init
coord(X) :- X = 0; X = 1; X = 2; X = 3; X = 4; X = 5.
coord(Y) :- Y = 0; Y = 1; Y = 2; Y = 3; Y = 4; Y = 5.
admissible_cells(X,Y) :- coord(X), coord(Y).


% Perception
nn(perc_net_version_1_NeSy_0,[Img, X,Y],Z,[human,nothing]) :: cell(Img,X,Y,Z).


% Prediction
attention_boundary(X,Y) :- X==3, \+(Y==5).
% attention_boundary(X,Y) :- X==1.

enlarged_boundary(X,Y) :- X==4, \+(Y==5).
enlarged_boundary(X,Y) :- X==5, \+(Y==5).
%enlarged_boundary(X,Y) :- X==4, \+(Y==5).


% Control
% 0 --> accelarate
% 1 --> break
% 2 --> idle

action(Img,1) :- admissible_cells(X,Y), attention_boundary(X,Y), cell(Img,X,Y, human).
action(Img,2)  :-  admissible_cells(X,Y), enlarged_boundary(X,Y), cell(Img,X,Y, human), \+ action(Img,1).
action(Img,0) :-  \+ action(Img,2), \+ action(Img,1).
