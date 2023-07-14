% Perception
nn(perc_net_version_2_NeSy_ped,[Img,Speed],X,[0,1,2,3]) :: cell_ped(Img,Speed,X).
nn(perc_net_version_2_NeSy_speed,[Img],C,[red,green,blue,black]) :: cell_c(Img,C).


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
    S < Zh.


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace


action(Img,Speed,1) :- cell_ped(Img,Speed,3);(cell_c(Img,C1), speed_zone_brake(C1,Speed)).
action(Img,Speed,2) :- cell_ped(Img,Speed,2), \+ action(Img,Speed,1).
action(Img,Speed,3) :- (cell_ped(Img,Speed,1);(cell_c(CImg,3), speed_zone_follow(C3,Speed))), \+ action(Img,Speed,1), \+ action(Img,Speed,2).
action(Img,Speed,0) :- cell_ped(Img,Speed,0),\+ action(Img,Speed,1), \+ action(Img,Speed,2), \+ action(Img,Speed,3).
