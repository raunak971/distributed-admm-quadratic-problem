using Distributed 
@everywhere begin 
  using PyPlot
  using LinearAlgebra
  using SharedArrays
  using Printf


  pygui(true) #so that plot appears in side window


  # generate H and h as required. (Works fine)
  function generate_H(n,s)
    #This function will generate H(n,n*s) and h(n,s) which will be
    #required and used as need. Note: This is only used to
    #generate and no calculations are done here
      
    H =Array{Float64}(undef, n, 0)
    h =Array{Float64}(undef, n, 0)

    for i=1:1:s     
      for_Hi = rand(Float64, (n, n))
      Hi = symposdef(for_Hi)
      H = hcat(H,Hi)
      hi = rand(Float64, (n, 1))
      h = hcat(h,hi)
    end
      
    return (H,h)
  end

  #ensuring that the matrix is semi positive definite matrix, if not this function will make it happen (Works fine)
  function symposdef(H)

    #In order to check if a matrix is symmetric positive definite matrix if
    # not then convert it into a symmetric positive definite matrix
      
      
    #FIrst test for a square matrix A
    r,c = size(H)
    if r != c
      println("A must be a square matrix.")
    elseif (r == 1) && (H <= 0)   
      # A was scalar and non-positive, so just return eps
      Hhat = eps()
      return
    end
    # symmetrize H into B
    B = (H + transpose(H))/2
    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    U,Sigma,V = svd(B)
    H = V*Diagonal(Sigma)*transpose(V)
    # get Hhat
    Hhat = (B+H)/2
    # ensure symmetry
    Hhat = (Hhat + transpose(Hhat))/2
    # test that Hhat is in fact PD. if it is not so, then tweak it just a bit.
    p = 1
    k = 0
    while p != 0
      k = k + 1
      R = try
        cholesky(Hhat)
        p = 0
      catch
        p = 1
      end 

      if p != 0
        mineig = minimum(eigvals(Hhat))
        Hhat = Hhat + (-mineig*k.^2 + eps(mineig))*I(size(H,1))
      end
    end
      return Hhat
  end

  #Cost function as mentioned in the exercise (Works fine)
  function cost_F(H, h, x)
    #---Just to compute value of our objective function------
    obj = 0.5*x'*H*x + h'*x    
    return obj   
  end


  #main function to solve using D-ADMM (Works fine)
  function solve_dadmm(H, h, rho)
    
    @everywhere function Imat(n)
      I_mat = Array{Float64}(undef,n,n);
      for i in 1:1:n
        for j in 1:1:n
          if i==j
            I_mat[i,j] = 1;
          else 
            I_mat[i,j] = 0;
          end
        end
      end
      return I_mat
    end

    @everywhere function cost_F(H, h, x)
      #---Just to compute value of our objective function------
      obj = 0.5*x'*H*x + h'*x;   
      return obj   
    end

    #=@everywhere function parallel_dadmm(i,col,k,x,y,z,objvali,rho,H,h,n,s)
      #global x, y, z, objvali;
      # x -update step
      x[:,i] = (H[:,col:(col+19)] + rho*Imat(n))\(rho*(z - y[:,i])-h[:,i]); 
      # global update for z
      z = sum(x+((1/rho)*y),dims=2)/s;              
      # y - update step
      y[:,i] = y[:,i] + rho*(x[:,i] - z);        
      # value of objective function
      objvali[k,i]  = cost_F(H[:,col:(col+19)], h[:,i], x[:,i]);
      return (x, y, z, objvali)
    end =#
      
    @everywhere function parallel_x(i,col,k,x,y,z,objvali,rho,H,h,n,s)
      # x -update step
      x[:,i] = (H[:,col:(col+19)] + rho*Imat(n))\(rho*(z - y[:,i])-h[:,i]);
      return x
    end

    @everywhere function parallel_y(i,col,k,x,y,z,objvali,rho,H,h,n,s)            
      # y - update step
      y[:,i] = y[:,i] + rho*(x[:,i] - z);
      return y
    end

    t_start = time();
    global z , x_star
    Output_to_screen= 0;
    MAX_ITER = 1000;
    
    n = size(H,1);
    s = size(h,2);
    x = convert(SharedArray,zeros(Float64,n,s));
    y = convert(SharedArray,zeros(Float64,n,s));

    sigma_H = zeros(Float64,n,n);
    for i = 1:n:(n*s)
      sigma_H = sigma_H + H[:,i:(i+19)];
    end
    
    objvali = convert(SharedArray,Array{Float64}(undef, MAX_ITER, s));
    objval = Array{Float64}(undef, MAX_ITER, 1);
    xk_zk_norm = Array{Float64}(undef, MAX_ITER, 1);
    xstar_xk_norm = Array{Float64}(undef, MAX_ITER, 1);
    x_sol =Array{Float64}(undef, n, 1);
    num_of_iter = 0;
    

    if Output_to_screen == 0
     @printf("%s\t%s\t%s\t%s\n", "iter","||xk-zk||2","||xstar-xk||2", "objective");
    end
    
    for k = 1:MAX_ITER
      num_of_iter += 1;      
      #= @sync for (worker,col) in zip(workers(),1:n:n*s)
        @async begin 
          x, y, z, objvali = remotecall_fetch(parallel_dadmm, worker, worker-1, col, k, x, y, z, objvali, rho, H, h, n, s);
        end
      end =#
      @sync for (worker,col) in zip(workers(),1:n:n*s)

        @async begin 
          x = remotecall_fetch(parallel_x, worker, worker-1, col, k, x, y, z, objvali, rho, H, h, n, s);
        end
        # global update for z
        z = sum(x+((1/rho)*y),dims=2)/s; 

        @async begin 
          y = remotecall_fetch(parallel_y, worker, worker-1, col, k, x, y, z, objvali, rho, H, h, n, s);
        end
      end
      
      
      # x - average
      x_sol = sum(x,dims=2)/s;
        
        
      objval[k] = cost_F(sigma_H,sum(h,dims=2),x_sol)[1];
      xk_zk_norm[k] = norm(x_sol -z)[1];
      xstar_xk_norm[k] = norm(x_sol - x_star)[1];
    
    
    
      if Output_to_screen == 0
        @printf("%d\t%f\t%f\t%f\n", k,xk_zk_norm[k],xstar_xk_norm[k], objval[k]);
      end
    
    
      if xk_zk_norm[k] < 1e-6
        break;
      end
    end
    
    if Output_to_screen == 0
      @printf("\n Elapsed time is %f",time()-t_start);
    end
    return (x_sol, objvali, objval, xk_zk_norm, xstar_xk_norm, num_of_iter)
  end
end  
  

# Part (1) --- Solving using KKT conditions -------

s = 10 #number of subsystems
n = 20 # dimensions of x

addprocs(s);

H =zeros(Float64,n,n*s);
global x_star
H,h = generate_H(n,s);


sigma_H = zeros(Float64,n,n);

for i = 1:n:(n*s)
  
  sigma_H = sigma_H + H[:,i:(i+19)];
end

sigma_h = sum(h,dims=2);

x_star = (sigma_H)\(-sigma_h);
OBJ_VAL = cost_F(sigma_H,sigma_h,x_star);

#Part (2) --- Solving using D-ADMM -----

rho = range(0.1; stop=2.3, length=12); # making empty arrays of type {Any} to hold values (in future will use struct)
x_sol = Array{Any}(undef, n, length(rho));
objvali = Array{Any}(undef, 1, length(rho));
objval = Array{Any}(undef, 1, length(rho));
xk_zk_norm = Array{Any}(undef, 1, length(rho));
xstar_xk_norm = Array{Any}(undef, 1, length(rho));
num_of_iter = Array{Any}(undef, length(rho),1);
ax1 = Array{Any}(undef,1, length(rho));
ax2 = Array{Any}(undef,1, length(rho));
ax3 = Array{Any}(undef,1, length(rho));

H = convert(SharedArray,H);
h = convert(SharedArray,h);


figure_1 = figure("Objective value vs Number of Iterations",figsize=(12,8)); 
ax1 = [subplot(4,3,i) for i=1:length(rho)];
figure_2 = figure("||xk - zk||2",figsize=(12,8));
ax2 = [subplot(4,3,i) for i=1:length(rho)];
figure_3 = figure("||xk - x*||2",figsize=(12,8));
ax3 = [subplot(4,3,i) for i=1:length(rho)];

#println("Which mode you want series(s) or parallel(s)?")
#exe_fashion = readline();

@time for i=1:1:length(rho)
  global z, x_sol,objvali, objval, xk_zk_norm, xstar_xk_norm, num_of_iter, ax1, ax2, ax3; #This will be broadcasted to all subsystems = {1,2,..,s} 
  z = convert(SharedArray,zeros(Float64,n,1));
  x_sol[:,i], objvali[i], objval[i], xk_zk_norm[i], xstar_xk_norm[i], num_of_iter[i] = solve_dadmm(H, h,rho[i]);
  
  ax1[i].plot(1:length(objval[i]),objval[i]);
  ax1[i].set_title(" ρ :" * string.(rho[i])); ax1[i].set_ylabel("f(x^k)"); ax1[i].set_xlabel("iter (k)");

  ax2[i].plot(1:length(objval[i]),xk_zk_norm[i]);
  ax2[i].set_title(" ρ :" * string.(rho[i])); ax2[i].set_ylabel("norm(xk - zk)"); ax2[i].set_xlabel("iter (k)");

  ax3[i].plot(1:length(objval[i]),xstar_xk_norm[i]);
  ax3[i].set_title(" ρ :" * string.(rho[i])); ax3[i].set_ylabel("norm(xstar - xk)"); ax3[i].set_xlabel("iter (k)");
  
end

figure_4 = figure(figsize=(12,8)); ax4 =gca();
ax4.plot(rho,num_of_iter, linestyle="dashdot", linewidth =4, marker = "+");
ax4.set_title("Effect of changing Value of ρ"); ax4.set_ylabel("Iterations required"); ax4.set_xlabel("ρ");

figure_1.savefig("Objective value vs Number of Iterations.png");
figure_2.savefig("norm(xk - zk).png");
figure_3.savefig("norm(xk - xstar).png");
figure_4.savefig("Impact of rho on Number of Iterations required.png"); 