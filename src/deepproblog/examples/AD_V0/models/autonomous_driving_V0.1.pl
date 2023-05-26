% Init
%coord(X) :- X = 0; X = 1; X = 2; X = 3; X = 4; X = 5; X = 6; X = 7; X = 8; X = 9.
%admissible_cells(X) :- coord(X).


% Perception
nn(perc_net_AD_V0,[Img],X,[0,1,2,3,4,5,6,7,8,9]) :: cell(Img,X).


% Prediction --> 80 pixels
attention_boundary(X) :- X==4.
attention_boundary(X) :- X==5.
attention_boundary(X) :- X==6.


% ---> 50 pixels
enlarged_boundary(X) :- X==7.
enlarged_boundary(X) :- X==8.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- cell(Img,X1), attention_boundary(X1).
action(Img,2)  :-  cell(Img,X2), enlarged_boundary(X2), \+ action(Img,1).
action(Img,0) :-  \+ action(Img,2), \+ action(Img,1).
