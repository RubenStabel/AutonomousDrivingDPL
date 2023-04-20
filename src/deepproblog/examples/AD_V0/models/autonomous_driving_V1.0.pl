% Init
coord(X) :- X = 0.
coord(X) :- X = 1.
coord(X) :- X = 2.
admissible_cells(X,Y) :- coord(X), coord(Y).


% Perception
nn(perc_net_AD_V1,[Img, X,Y],Z,[human,nothing]) :: cell(Img,X,Y,Z).


% Prediction
attention_boundary(X,Y) :- X==0.
attention_boundary(X,Y) :- X==1.

enlarged_boundary(X,Y) :- X==2.


% Control
% 0 --> accelarate
% 1 --> break
% 2 --> idle

action(Img,1) :- admissible_cells(X,Y), attention_boundary(X,Y), cell(Img,X,Y, human).
action(Img,2)  :-  admissible_cells(X,Y), enlarged_boundary(X,Y), cell(Img,X,Y, human), \+ action(Img,1).
action(Img,0) :-  \+ action(Img,2), \+ action(Img,1).
