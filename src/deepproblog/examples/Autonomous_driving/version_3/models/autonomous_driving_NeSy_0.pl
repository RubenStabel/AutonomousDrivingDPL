% Perception
nn(perc_net_version_3_NeSy_danger_pedestrian,[Img,Speed],X,[0,1,2,3]) :: cell_danger_pedestrian(Img,Speed,X).
nn(perc_net_version_3_NeSy_speed_zone,[MNIST],SZ,[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]) :: cell_speed_zone(MNIST,SZ).
nn(perc_net_version_3_NeSy_traffic_light,[Img],X,[green, orange, red]) :: cell_traffic_light(Img,X).
nn(perc_net_version_3_NeSy_danger,[Img,Speed],X,[0,2,3]) :: cell_danger(Img,Speed,X).


% Rules - speed zone
speed_zone_brake(SZ,S) :-
    S > SZ.

speed_zone_follow(SZ,S) :-
    round(S) == SZ.


% Rules - traffic light
traffic_light_brake(red,3,S) :- S > 0.2.
traffic_light_brake(orange,3,S) :- S > 0.2.

traffic_light_idle(red,3,S) :- S =< 0.2.
traffic_light_idle(orange,3,S) :- S =< 0.2.
traffic_light_idle(orange,2,S).

traffic_light_accelerate(green,3).
traffic_light_accelerate(green,2).
traffic_light_accelerate(green,0).
traffic_light_accelerate(orange,0).
traffic_light_accelerate(red,0).


% Control
% 0 --> accelerate
% 1 --> break
% 2 --> idle
% 3 --> keep pace

action(Img,MNIST,Speed,1) :-
    cell_danger_pedestrian(Img,Speed,3);
    (cell_speed_zone(MNIST,SZ1), speed_zone_brake(SZ1,Speed));
    (cell_traffic_light(Img,C), cell_danger(Img,Speed,D), traffic_light_brake(C,D,Speed)).
action(Img,MNIST,Speed,2) :-
    (cell_danger_pedestrian(Img,Speed,2);
    (cell_traffic_light(Img,C), cell_danger(Img,Speed,D), traffic_light_idle(C,D,Speed))),
    \+ action(Img,MNIST,Speed,1).
action(Img,MNIST,Speed,3) :-
    (cell_danger_pedestrian(Img,Speed,1);
    (cell_speed_zone(MNIST,SZ3), speed_zone_follow(S3,Speed))),
    \+ action(Img,MNIST,Speed,1),
    \+ action(Img,MNIST,Speed,2).
action(Img,MNIST,Speed,0) :-
    (cell_danger_pedestrian(Img,Speed,0);
    (cell_traffic_light(Img,C), cell_danger(Img,Speed,D), traffic_light_accelerate(C,D))),
    \+ action(Img,MNIST,Speed,1),
    \+ action(Img,MNIST,Speed,2),
    \+ action(Img,MNIST,Speed,3).
