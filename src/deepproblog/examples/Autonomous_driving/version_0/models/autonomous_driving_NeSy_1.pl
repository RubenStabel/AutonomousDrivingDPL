% Perception
nn(perc_net_version_0_NeSy_1,[Img],X,[0,1,2,3,4,5,6,7,8,9]) :: cell(Img,X).


% Prediction --> 80 pixels
attention_boundary(X) :- X==4.
attention_boundary(X) :- X==5.
attention_boundary(X) :- X==6.


% Prediction ---> 50 pixels
enlarged_boundary(X) :- X==7.
enlarged_boundary(X) :- X==8.

% Prediction ---> ... pixels
out_of_view(X) :- X==0; X==1; X==2; X==3; X==9.

% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- cell(Img,X1), attention_boundary(X1).
action(Img,2)  :-  cell(Img,X2), enlarged_boundary(X2), \+ action(Img,1).
action(Img,0) :-  cell(Img,X3), out_of_view(X3), \+ action(Img,2), \+ action(Img,1).
