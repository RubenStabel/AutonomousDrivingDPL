% Perception
nn(perc_net_AD_V1X,[Img],X,[0,1,2,3,4,5,6,7,8,9]) :: cell_x(Img,X).
nn(perc_net_AD_V1Y,[Img],Y,[0,1]) :: cell_y(Img,Y).


% Prediction --> 80 pixels
attention_boundary(X,y) :- X==4, Y=0.
attention_boundary(X,Y) :- X==5, Y=0.
attention_boundary(X,Y) :- X==6, Y=0.


% Prediction ---> 50 pixels
enlarged_boundary(X,Y) :- X==7, Y=0.
enlarged_boundary(X,Y) :- X==8, Y=0.

% Prediction ---> ... pixels
out_of_view(X,Y) :- X==0; X==1; X==2; X==3; X==9; Y==1.

% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle

action(Img,1) :- cell_x(Img,X1), cell_y(Img,Y1), attention_boundary(X1,Y1).
action(Img,2)  :-  cell_x(Img,X2), cell_y(Img,Y2), enlarged_boundary(X2,Y2), \+ action(Img,1).
action(Img,0) :-  cell_x(Img,X3), cell_y(Img,Y3), out_of_view(X3,Y3), \+ action(Img,2), \+ action(Img,1).
