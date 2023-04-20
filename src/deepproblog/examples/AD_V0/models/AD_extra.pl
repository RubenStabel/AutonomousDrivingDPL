nn(perc_net,[Img, X,Y],Z,[human,nothing]) :: cell(Img,X,Y,Z).

autonomous_driving(Img,X,Y,Z) :- cell(Img,X,Y,Z).

attention_boundary(X,Y) :-
    X == 3 ; X == 4,
    Y1 is Yc-Y,  % Relative position so not nec
    Y =< 2.
enlarged_boundary(X,Y,Xc,Yc) :-
    X1 is X-1,
    Y1 is Y+1,
    Y2 is Y-1,
    0 =< X1,
    X1 =< 5,
    Y1 =< 5,
    Y2 >= 0,
    attention_boundary(X1,Y1,Xc,Yc) ; attention_boundary(X1,Y,Xc,Yc) ; attention_boundary(X1,Y2,Xc,Yc).


occluded(Img1,Img2,Xc,Yc) :-
    cell(Img1,X,Y,human),
    enlarged_boundary(X,Y,Xc,Yc),
    X1 is X+1,
    X2 is X-1,
    Y1 is Y+1,
    Y2 is Y-1,
    cell(Img2,X,Y,nothing), cell(Img2,X1,Y1,nothing), cell(Img2,X1,Y2,nothing), cell(Img2,X2,Y1,nothing), cell(Img2,X2,Y2,nothing).


action(Img1,Img2,Xc,Yc,Vel,break) :-
    attention_boundary(X,Y,Xc,Yc),
    cell(Img2,X,Y,human),
    Vel > 0.
action(Img1,Img2,Xc,Yc,Vel,idle) :-
    cell(Img1,X,Y,human),
    enlarged_boundary(X,Y),
    Vel >= 4.
action(Img1,Img2,Xc,Yc,Vel,accelerate).







action(Img1,Img2,Xc,Yc,Vel,break) :-
    occluded(Img1,Img2, Yc),
    Vel >= 4.
action(Img1,Img2,Xc,Yc,Vel,idle) :-
    cell(Img2,X,Y,human),
    attention_boundary(X,Y,Xc,Yc),
    Vel == 0.