function [grad_opt, b_val, phaseFinal] = opt_gradient_waveform(alg,CGs,MMT,grad,T_ECHO,bvalue_target,dt,G_Max,S_Max,Gvec,T_90,T_RF)
    %Manuscript: Optimized Diffusion-Weighting Gradient Waveform Design 
    %(ODGD) Formulation for Motion Compensation and Concomitant Gradient
    %Nulling. Magnetic Resonance in Medicine. 2018.
    %Oscar Pena-Nogales, Yuxin Zhang, Xiaoke Wang, Rodrigo de Luis-Garcia,
    %Santiago Aja-Fernandez and Diego Hernando.     
    %
    %
    %This function computes the ODGD diffusion-weighting gradient waveforms
    %for a target TE with maximum b-value subject to gradient hardware constraints, moment
    %nulling constraints, sequence timing constraints, and/or concomitant
    %gradient nulling constraints. 
    %
    %
    % INPUTS  : alg            - ODGD (1) or CODE (0) 
    %           CGs            - Conocmitant Gradients-nulling (1/0)
    %           MMT            - Desired waveform moments
    %           grad            - initial gradient. 
    %           T_ECHO        - EPI time to Echo [ms]
    %           bvalue_target     - Desired b-value [s/mm2] if it is 0 it
    %                               does not derate the waveform.
    %           dt              - time resolution.
    %           G_Max           - Max Gradient amplitude [T/m]   
    %           S_Max           - Maximum gradient slew rate [T/m/s]
    %           Gvec            - Diffusion encoding vector magnitude: sqrt(Gx^2 + Gy^2 + Gz^2)
    %           T_90            - Start time of diffusion encoding [ms]
    %           T_RF            - - Refocusing pulse duration [ms]
    %
    % OUTPUTS : grad_opt - Final ODGD or CODE gradient waveform
    %            b_val - Final b-value   
    %            phaseFinal - remaining phase of the grad_opt waveform  
    
    
    % Laboratorio de Procesado de Imagen - Universidad de Valladolid, Spain
    % Departments of Medical Physics, Radiology, and Biomedical
    % Engineering, University of Wisconsin-Madison, WI, USA.
    % - Oscar Pe?na Nogales (opennog@lpi.tel.uva.es)
    % - Yuxin Zhang (yzhang785@wisc.edu)
    % - Xiaoke Wang 
    % - Rodrigo de Luis-Garcia 
    % - Santiago Aja-Fernandez
    % - James H. Holmes
    % - Diego Hernando (dhernando@wisc.edu)
    % - May 1, 2018

%For the first step we start with their optimization and we dont modify the
%different timings.
    if nargin==0 || nargin==1 || nargin==2 || nargin==3 || nargin==4 || nargin==5 || nargin==6  || nargin==7 || nargin==8 
        G_Max = 49e-3;          % T/m
        Gvec = sqrt(1);           % magnitude of direction vector (sqrt(Gx^2 + Gy^2 + Gz^2)). For example, Gvec = 1 for only x encoding
        S_Max = 100;             % T/m/s
        T_90 = 5.3;             % Start time of diffusion. Typically the duration of excitation + EPI correction lines [ms]
        T_RF = 4.3;             % 180 duration. [ms]
                        
        if nargin==0
            alg = 1; %alg to select the funtion to optimize (objective funtion), 1 for the real bValue and 0 for the pseudobvalue
        end
        if nargin<2
            CGs = 0; %if we want to take the conocmitant gradients into account: 1-YES, 0-NO
        end
        if nargin<3
            MMT = 0; 
        end
        if nargin<5
            grad=zeros(100,1);
        end
        if nargin<6
            T_ECHO   = 26.4;        % EPI time to center k-space line [ms]
        end
        if nargin<7
            bvalue_target = 0;        % s/mm2
        end
        if nargin<8
            dt = 0.5e-3;              % timestep of optimization [s] (increase for faster simulation)
        end
        
        % Hardware constraints
        G_Max = G_Max*Gvec;
        S_Max = S_Max*Gvec;          % T/m/s
    end
 
    %% Define some constants
    % Physical constants
    GAMMA = 42.58e3;          % Hz/mT for 1H (protons)
    
    %% Define the moment nulling vector
    switch MMT
      case 0
        mvec =  [0];          % M0 nulled gradients
      case 1
        mvec =  [0,0];        % M0+M1 nulled gradients
      case 2
        mvec =  [0,0,0];      % M0+M1+M2 nulled gradients
    end
    
    n = numel(grad);
    tDiff=n*dt/(1e-3);
    
    ADCcont = ceil(T_ECHO*1e-3/dt)*dt/(1e-3); %EPI time to the center of the k-space
    preTime = ceil(T_90*1e-3/dt)*dt/(1e-3); %end of RF90.
    RFTime = ceil(T_RF*1e-3/dt)*dt/(1e-3);
    
    tECHO = n + ADCcont/(dt*1e3);
    
    tINV = floor(tECHO/2);
    INV = ones(n,1);   INV(tINV:end) = -1;
    C=tril(ones(n));
    C2 = C'*C;
    
    if size(mvec,2)>size(mvec,1)
        mvec = mvec';
    end
    
    t0 = 0;
    tf = 0;
  
    %Seed
    start=grad;
    
    %% Constraints
    %Upper/Lower Bound -> Amplitude constraint
    ub=G_Max*ones(n,1);
    ub(1:floor(preTime/(dt*1e3)))=0;
    ub(tINV-floor(RFTime/(dt*1e3)/2):tINV+ceil(RFTime/(dt*1e3)/2))=0; 
    ub(end)=0;
    lb=-G_Max*ones(n,1);
    lb(1:floor(preTime/(dt*1e3)))=0;
    lb(tINV-floor(RFTime/(dt*1e3)/2):tINV+ceil(RFTime/(dt*1e3)/2))=0;
    lb(end)=0;
    
    %Inequality constraint-> Slew rate
    D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
    D = D(1:end-1,:)/dt;
    
    A=zeros(2*size(D,1),n);
    A(1:size(D,1),:)=D;
    A(size(D,1)+1:end,:)=-D;
    
    b=S_Max*ones(size(A,1),1);
    
    %Equality constraint->Nulling at some parts
    %For the RF90,RF180 and ending.
    Aeq=zeros(n,1);
    Aeq(1)=1;
    Aeq(1:floor(preTime/(dt*1e3)))=1;
    Aeq(tINV-floor(RFTime/(dt*1e3)/2):tINV+ceil(RFTime/(dt*1e3)/2))=1;
    Aeq(end)=1;

    %Moments.
    Nm = size(mvec, 1);
    tvec = t0 + [0:n-1]*dt; % time vector [s]

    tMat = zeros( Nm, n );
    for mm=1:Nm,
      tMat( mm, : ) = tvec.^(mm-1);
    end
    
    aux=zeros(n+Nm,n);
    aux(1:n,:)=diag(Aeq);
    aux(n+1:n+Nm,:)=GAMMA*dt*(tMat.*repmat(INV',[size(tMat,1),1])); %ones(1,100).*INV';
    Aeq=aux;

    %finally we set the equality value:
    sAeq=size(Aeq);
    beq=zeros(sAeq(1),1); 
    
    %Non-linear Constraints -> CGs
    paramPhase={CGs,ones(1,n),dt,INV};
    nonlcon = @(x) constraints(x,paramPhase);
    
    %% Objective function
    if ~alg
        fun =@(x) -sum(cumsum(C*x)); % CODE
    else
        fun =@(x) -(GAMMA*2*pi)^2*(x.*INV)'*(C2*(x.*INV))*dt^3; %ODGD
    end

    %% Solver
    options = optimset( 'Algorithm','sqp','Display', 'off', 'GradObj', 'off', ...
                    'DerivativeCheck', 'on', 'MaxIter', ...
                    0.5e3, 'MaxFunEvals', 1e10 , 'LargeScale', ...
                    'on','TolFun',1e-17,'TolX',1e-20,'TolCon',1e-8);

    if CGs
        [value,fval,exitflag,ouput] = fmincon(fun, start,A,b,Aeq,beq,lb,ub,nonlcon,options);
    else
        [value,fval,exitflag,ouput] = fmincon(fun, start,A,b,Aeq,beq,lb,ub,[],options);
    end

    grad_opt=value;
    b_val = (GAMMA*2*pi)^2*(value.*INV*dt)'*(C2*(value.*INV*dt))*dt;
    
    %if we want a target b-value.
    if bvalue_target~=0 && bvalue_target<b_val
        [value, b_val] = derating(value,bvalue_target,INV,dt,C2,GAMMA);
        grad_opt=value;
    end
    
    %% Check results
    % form time vector to calculate moments
    tvec = t0 + (0:n-1)*dt; % in sec
    tMat = zeros( 3, n );
    for mm=1:3,
      tMat( mm, : ) = tvec.^(mm-1);
    end
    
    % progressive vectors for m0, m1, m2
    tMat0 = tril(ones(n)).*repmat(tMat(1,:)',[1,n])';
    tMat1 = tril(ones(n)).*repmat(tMat(2,:)',[1,n])';
    tMat2 = tril(ones(n)).*repmat(tMat(3,:)',[1,n])';
    
    % final moments
    moments = GAMMA*dt*tMat*(value.*INV); 
    phaseFinal = dt*tMat(1,:)*(value.^2.*INV); %%We null: ceq=dt*tMat*((x.^2).*INV); 
    
    % moments and phase over time
    M0 = GAMMA*dt*tMat0*(value.*INV);
    M1 = GAMMA*dt*tMat1*(value.*INV);
    M2 = GAMMA*dt*tMat2*(value.*INV);
    phaseFinal = dt*tMat0*(value.^2.*INV); 
       
    % final b-value
    b_val = (GAMMA*2*pi)^2*(value.*INV*dt)'*(C2*(value.*INV*dt))*dt;
    
    % diffusion encoding duration
    tDiff = length(grad)*dt/(1e-3);

    TE = tDiff + ADCcont;
    
    D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
    D = D(1:end-1,:)/dt;
    
    DESCRIPTION = ['bValue: ' num2str(round(b_val)) ', TE: ' num2str(TE) '---Alg: ' num2str(alg)   '---SRmax: ' num2str(max(abs(D*grad_opt/Gvec)))];
    
    %% Generate a figure
%     figure; subplot(4,1,1);
%     plot(value,'LineWidth',2);
%     title(DESCRIPTION); ylabel('G waveform');
%     subplot(4,1,2);
%     plot(M1/100,'r','LineWidth',2); hold on; plot(M2,'LineWidth',2);ylabel('Moments');
%     legend('m1','m2','location','northwest');
%     subplot(4,1,3);
%     plot(abs(D*value/Gvec),'LineWidth',2);ylabel('dG/dt'); %slew rate
%     subplot(4,1,4);
%     plot(phaseFinal,'LineWidth',2);ylabel('phi');
    
    phaseFinal=phaseFinal(end);
    
end

function [c, ceq]=constraints(x,parPhase)
    %phase
    CGs=parPhase{1};
    tMat=parPhase{2};
    dt=parPhase{3};
    INV=parPhase{4};
       
    %constraints
    c=[];
    ceq=dt*tMat*((x.^2).*INV); %its enought to null the dephase when considering any gradient combination WHILE they all have the same waveform. 
end


function [Gtmp, b_val] = derating(grad,bvalue_target,INV,dt,C2,GAMMA)
    scale=[1:-0.00005:0.00005]; 
    done=0;
    i=1;
    while (done==0)
        Gtmp=grad*scale(i);
        b_val = (GAMMA*2*pi)^2*(Gtmp.*INV*dt)'*(C2*(Gtmp.*INV*dt))*dt;
        if b_val<=bvalue_target
            done=1;
        end
        i = i+1;
        if i>size(scale,2)
            done=1;
        end
    end

end












