% Init
coord(X) :- X = 0; X = 1; X = 2; X = 3.
admissible_cells(X) :- coord(X).


% Perception
nn(perc_net_AD_V1,[Img, X],Z,[human,nothing]) :: cell(Img,X,Z).


% Prediction
attention_boundary(X) :- X==1.
%attention_boundary(X) :- X==7.
%attention_boundary(X) :- X==8.

enlarged_boundary(X) :- X==1.
enlarged_boundary(X) :- X==2.
enlarged_boundary(X) :- X==3.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- admissible_cells(X), attention_boundary(X), cell(Img,X, human).
action(Img,2)  :-  admissible_cells(X), enlarged_boundary(X), cell(Img,X, human), \+ action(Img,1).
action(Img,0) :-  \+ action(Img,2), \+ action(Img,1).
