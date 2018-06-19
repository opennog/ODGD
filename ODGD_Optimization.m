function [grad_opt, b_val]=ODGD_Optimization(bvalue_T,alg,MMT,T_ECHO,CGs)
    %Manuscript: Optimized Diffusion-Weighting Gradient Waveform Design 
    %(ODGD) Formulation for Motion Compensation and Concomitant Gradient
    %Nulling. Magnetic Resonance in Medicine. 2018.
    %Oscar Pena-Nogales, Yuxin Zhang, Xiaoke Wang, Rodrigo de Luis-Garcia,
    %Santiago Aja-Fernandez and Diego Hernando. 
    %
    %
    %This function computes the ODGD diffusion-weighting gradient waveforms
    %for a target b-value with minimum TE subject to gradient hardware constraints, moment
    %nulling constraints, sequence timing constraints, and/or concomitant
    %gradient nulling constraints. 
    %
    %
    % INPUTS  : bvalue_T     - Desired b-value [s/mm2] if it is 0 it
    %                               does not derate the waveform.
    %           alg            - ODGD (1) or CODE (0) 
    %           MMT            - Desired waveform moments
    %           T_ECHO        - EPI time to Echo [ms]
    %           CGs            - Conocmitant Gradients-nulling (1/0)
    %
    % OUTPUTS : grad_opt - Final ODGD gradient waveform
    %            b_val - Final b-value     
    
    % Laboratorio de Procesado de Imagen (www.lpi.te.uva.es) - Universidad de Valladolid, Spain
    % Departments of Medical Physics, Radiology, and Biomedical
    % Engineering, University of Wisconsin-Madison, WI, USA.
    % - Oscar Pena Nogales (opennog@lpi.tel.uva.es)
    % - Yuxin Zhang (yzhang785@wisc.edu)
    % - Xiaoke Wang 
    % - Rodrigo de Luis-Garcia 
    % - Santiago Aja-Fernandez
    % - James H. Holmes
    % - Diego Hernando (dhernando@wisc.edu)
    % - May 1, 2018

%% Define default values to generate a test waveform
    if nargin==0 || nargin==1 || nargin==2 || nargin==3 || nargin==4 || nargin==5
        G_Max     = 49e-3;            % T/m
        Gvec      = sqrt(1+0+0);      % magnitude of direction vector (sqrt(Gx^2 + Gy^2 + Gz^2)). For example, Gvec = 1 for only x encoding
        S_Max     = 100;               % T/m/s
        T_90      = 5.3;              % Start time of diffusion. Typically the duration of excitation + EPI correction lines [ms]
        T_RF      = 4.3;              % 180 duration. [ms]
        
        if nargin==0
            bvalue_T    = 1000;              % target b-value s/mm2
        end
        if nargin<2
            alg=1; %alg to select the funtion to optimize (objective funtion), 1 ODGD and 0 CODE
        end
        if nargin<3
            MMT= 0;                % Desired waveform moments- [0 for M0=0, 1 for M0=M1=0, 2 for M0=M1=M2=0]
        end
        if nargin<4
            T_ECHO    = 26.4;             % EPI time to center k-space line [ms]
        end
        if nargin<5
            CGs=0; %if we want to take the conocmitant gradients into account: 1-YES, 0-NO
        end
    end

    % Simulation constants
    dt      = 0.5e-3;              % timestep of optimization [s] (increase for faster simulation)
    GAMMA = 42.58e3;
    G_Max = G_Max*Gvec;
    S_Max = S_Max*Gvec;          % T/m/s
    
    ADCcont = ceil(T_ECHO*1e-3/dt)*dt/(1e-3);
    
    %% Design the upper-bound symmetric gradient waveforms
    n_top = upper_bound(bvalue_T, MMT, T_ECHO, G_Max, S_Max, Gvec, T_90, T_RF, dt)+ADCcont/(dt*1e3);
%     if alg==0 && MMT==2
%         n_top = n_top + 11; %this is needed because in many cases CODE M2 works worse than the traditional waveforms 
%     end

    
    %% Define time and index bounds
    %IMPORTANT: To compute the optimal waveform with CG nulling, it assumes
    %that the same waveform without the nulling has been previously
    %computed, and saved accordingly.
    if CGs==0
        tLow  = 2*(T_RF/2 + ADCcont);    % TE of SE (b=0) sequence
        n_bot = floor(tLow * 1e-3/dt);  % lower bound on TE 
    else
        %We could also compute the optimal waveform for CG=0 and
        %then modify this code just to load that waveform.
        grad_opt=0;
        load('grad_opt.mat','grad_opt');
        n_bot=numel(grad_opt)+ADCcont/(dt*1e3);
        clear grad_opt
    end

    %% Run the optimization
    clc;
    fprintf(strcat('Optimizing...... b-value: ',num2str(bvalue_T),' MMT = ',num2str(MMT),' CGs = ',num2str(CGs),' \n'));
    done = 0; n = n_top; iter=0; b_val=0; 
    while(done==0)
        fprintf('..... TE <= %2.1fms ... Gap: %2.1fms \n' ,n_top*dt/(1e-3), (n_top-n_bot)*dt/(1e-3));
        iter=iter+1;
        n=ceil((n_top+n_bot)/2)-ADCcont/(dt*1e3);
        if CGs==0
            [grad_opt, b_val] = opt_gradient_waveform(alg,0,MMT,ones(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
        elseif CGs==1 %the seed for CG nulling is the one waveform without nulling
                [tmpgrad, tmpb_val] = opt_gradient_waveform(alg,0,MMT,ones(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                [grad_opt, b_val, phaseFinal] = opt_gradient_waveform(alg,CGs,MMT,tmpgrad,T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                if phaseFinal>1e-8 %In case the phase is not nulled.
                    disp('Phase not nulled.');
                    [grad_opt, b_val, phaseFinal] = opt_gradient_waveform(alg,CGs,MMT,zeros(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                end
        end
        
        if bvalue_T<=ceil(b_val) 
            n_top=n+ADCcont* 1e-3/dt;
        elseif bvalue_T>ceil(b_val) 
            n_bot=n+ADCcont* 1e-3/dt;
        end

        %FINAL
        if n_top-n_bot<2 
            n=ceil((n_top+n_bot)/2)-ADCcont/(dt*1e3);
            fprintf('FINAL: TE = %2.1fms ... \n' ,n*dt/(1e-3)+ADCcont);
            if CGs==0
                [grad_opt, b_val] = opt_gradient_waveform(alg,CGs,MMT,ones(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
            else
                [tmpgrad, tmpb_val] = opt_gradient_waveform(alg,0,MMT,ones(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                [grad_opt, b_val, phaseFinal] = opt_gradient_waveform(alg,CGs,MMT,tmpgrad,T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                if phaseFinal>1e-8 %In case the phase is not nulled.
                    disp('Phase not nulled.');
                    [grad_opt, b_val, phaseFinal] = opt_gradient_waveform(alg,CGs,MMT,zeros(n,1),T_ECHO,0,dt,G_Max,S_Max,Gvec,T_90,T_RF);
                end
            end
            
            if ceil(b_val)>=bvalue_T
                fprintf('Derating... b-val: %d, TE=%2.1fms\n',round(b_val),n*dt*1e3+ADCcont);
                tECHO = n + ADCcont/(dt*1e3);
                tINV = floor(tECHO/2);
                INV = ones(n,1);   INV(tINV:end) = -1;
                C=tril(ones(n));
                C2 = C'*C;
                [grad_opt, b_val] = derating(grad_opt,bvalue_T,INV,dt,C2,GAMMA);
                done=1;
            else
                fprintf('ERROR..... extending TE \n');
                n_bot=n+ADCcont*1e-3/dt;
                n_top=n_top+40;
                done=0;
            end
            
        end
        
    end
    
    %% Check results
    % form time vector to calculate moments
    n=length(grad_opt);
    
    t0=0;
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
    moments = GAMMA*dt*tMat*(grad_opt.*INV); 
    phaseFinal = dt*tMat(1,:)*(grad_opt.^2.*INV);
    
    % moments and phase over time
    M0 = GAMMA*dt*tMat0*(grad_opt.*INV);
    M1 = GAMMA*dt*tMat1*(grad_opt.*INV);
    M2 = GAMMA*dt*tMat2*(grad_opt.*INV);
    phaseFinal = dt*tMat0*(grad_opt.^2.*INV);
    
    %final Eddy Currents
    D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
    D = D(1:end-1,:)/dt*grad_opt;
    D(end+1)=0;
    
    % final b-value
    b_val = (GAMMA*2*pi)^2*(grad_opt.*INV*dt)'*(C2*(grad_opt.*INV*dt))*dt;
    
    % diffusion encoding duration
    tDiff = n*dt/(1e-3);

    TE = tDiff + ADCcont;
    
    D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
    D = D(1:end-1,:)/dt;
    
    DESCRIPTION = ['bValue: ' num2str(round(b_val)) ', TE: ' num2str(TE) ];
        
    %%Save waveform
    save('grad_opt.mat','grad_opt');
    
    %% Generate a figure
    figure; subplot(3,1,1);
    plot(grad_opt,'LineWidth',2);
    title(DESCRIPTION); ylabel('G waveform');
    subplot(3,1,2);
    plot(M1/100,'r','LineWidth',2); hold on; plot(M2,'LineWidth',2);ylabel('Moments');
    legend('m1','m2','location','northwest');
    subplot(3,1,3);
    plot(phaseFinal,'LineWidth',2);ylabel('phi');

end

function n_top =upper_bound(bvalue_T, MMT, T_ECHO, G_Max, S_Max, Gvec, T_90, T_RF,dt)

    b_val=-1;
    n=60;
    if MMT==0
        while(bvalue_T>=ceil(b_val))
            n=n+1;
            [G_MONO, b_val]=MONO(zeros(n,1), T_ECHO, 0, dt, G_Max, S_Max, Gvec, T_90, T_RF); %0 to avoid the loop
        end
        n_top=n;
    elseif MMT==1
        while(bvalue_T>=ceil(b_val))
            n=n+1;
            [G_BIPOLAR, b_val]=BIPOLAR(zeros(n,1), T_ECHO, 0,dt, G_Max, S_Max, Gvec, T_90, T_RF); %0 to avoid the loop
        end
        n_top=n;
    elseif MMT==2
        while(bvalue_T>=ceil(b_val))
            n=n+1;
            [G_MOCO, b_val]=MOCO(zeros(n,1), T_ECHO, 0, dt, G_Max, S_Max, Gvec, T_90, T_RF); %0 to avoid the loop
        end
        n_top=n;
    end
end

function [Gtmp, b_val] = derating(grad,bvalue_target,INV,dt,C2,GAMMA)
    scale=[1:-0.0005:0.0005]; 
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