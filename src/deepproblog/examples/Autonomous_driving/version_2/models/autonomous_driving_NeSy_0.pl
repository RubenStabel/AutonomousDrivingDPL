% Perception
nn(perc_net_version_2_NeSy_x,[Img],X,[0,1,2,3]) :: cell_x(Img,X).
nn(perc_net_version_2_NeSy_y,[Img],Y,[0,1,2]) :: cell_y(Img,Y).
nn(perc_net_version_2_NeSy_c,[Img],C,[red,green,blue,black]) :: cell_c(Img,C).


% Prediction --> 80 pixels
attention_boundary(X,Y) :- X==1, Y=0.


% Prediction ---> 50 pixels
enlarged_boundary(X,Y) :- X==1, Y=1.
enlarged_boundary(X,Y) :- X==2, Y=0.
enlarged_boundary(X,Y) :- X==2, Y=1.


% Prediction ---> ... pixels
out_of_view(X,Y) :- X==0; X==3; Y==2.


% Speed zone actions
map_c_s(red,2.0).
map_c_s(green,4.0).
map_c_s(blue,6.0).
map_c_s(black,8.0).

speed_zone_brake(C,S) :-
    map_c_s(C,Z),
    S > Z.

speed_zone_follow(C,S) :-
    map_c_s(C,Z),
    Zl is Z - 0.1,
    Zh is Z + 0.1,
    Zl < S,
    S <Zh.

% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,Speed,1) :- cell_x(Img,X1), cell_y(Img,Y1), cell_c(Img,C1), (attention_boundary(X1,Y1);speed_zone_brake(C1,Speed)).
action(Img,Speed,2) :- cell_x(Img,X2), cell_y(Img,Y2), enlarged_boundary(X2,Y2), \+ action(Img,Speed,1).
action(Img,Speed,3) :- cell_c(Img,C3), speed_zone_follow(C3,Speed), \+ action(Img,Speed,1), \+ action(Img,Speed,2).
action(Img,Speed,0) :- cell_x(Img,X3), cell_y(Img,Y3), out_of_view(X3,Y3), \+ action(Img,Speed,1), \+ action(Img,Speed,2), \+ action(Img,Speed,3).
