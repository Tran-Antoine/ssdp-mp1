function  R = right_dual(c,m) % Kx1
 R = zeros(2*m-1,m);
 
for i = 0:m-1
    R(i+1:i+1+m-1,i+1) =  c;
end


    
end