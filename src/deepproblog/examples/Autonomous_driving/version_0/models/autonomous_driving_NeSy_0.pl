% Init
%; X = 5.0; X = 6.0; X = 7.0; X = 8.0; X = 9.0
coord(X) :- X = 0.0; X = 1.0; X = 2.0; X = 3.0; X = 4.0.
admissible_cells(X) :- coord(X).


% Perception
nn(perc_net_version_0_NeSy_0,[Img, X],Z,[human,nothing]) :: cell(Img,X,Z).


% Prediction --> 80 pixels
attention_boundary(X) :- X==1.0.


% ---> 50 pixels
enlarged_boundary(X) :- X==2.0.
enlarged_boundary(X) :- X==3.0.
%enlarged_boundary(X) :- X==4.0.


% Prediction ---> ... pixels
out_of_view(X) :- X==0.0; X==4.0.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- admissible_cells(X), attention_boundary(X), cell(Img,X, human).
action(Img,2)  :-  admissible_cells(X), enlarged_boundary(X), cell(Img,X,human), \+ action(Img,1).
action(Img,0) :-  admissible_cells(X), out_of_view(X), \+ action(Img,2), \+ action(Img,1).
