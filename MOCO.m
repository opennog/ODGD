function [G_MOCO, b_val, phaseFinal]=MOCO(grad,T_ECHO,bvalue_target,dt,G_Max,S_Max,Gvec,T_90,T_RF)
    %Manuscript: Optimized Diffusion-Weighting Gradient Waveform Design 
    %(ODGD) Formulation for Motion Compensation and Concomitant Gradient
    %Nulling. Magnetic Resonance in Medicine. 2018.
    %Oscar Pena-Nogales, Yuxin Zhang, Xiaoke Wang, Rodrigo de Luis-Garcia,
    %Santiago Aja-Fernandez and Diego Hernando. 
    %
    %
    %This function computes the MOCO gradient waveform and its bValue, and
    %the final phase.
    %
    % INPUTS  : grad            - initial gradient. Only needed to set the
    %                               TE of the waveform. 
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
    % OUTPUTS : G_MOCO - Final MOCO gradient waveform
    %            b_val - Final b-value   
    %            phaseFinal - remaining phase of the G_MOCO waveform
    
    
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

    if nargin==0 || nargin==1 || nargin==2 || nargin==3 || nargin==4
      G_Max = 49e-3;          % T/m
      Gvec = sqrt(1);           % magnitude of direction vector (sqrt(Gx^2 + Gy^2 + Gz^2)). For example, Gvec = 1 for only x encoding
      S_Max = 100;             % T/m/s
      T_90 = 5.3;             % Start time of diffusion. Typically the duration of excitation + EPI correction lines [ms]
      T_RF = 4.3;             % 180 duration. [ms]
      
      if nargin==0
        grad=zeros(201,1);
        
      end
      if nargin<2
          T_ECHO   = 26.4;        % EPI time to center k-space line [ms]
      end
      if nargin<3
          bvalue_target = 00;    % s/mm2 if bvalue_target==0 it does not derate the waveform
      end
      if nargin<4
          dt = 0.5e-3;              % timestep of optimization [s] (increase for faster simulation)
      end
    end

    % modified MOCO param
    %This code can provide the original MOCO waveforms defined
    %by Chirstian T. Stoeck in 2016 'Second order motion compensated spin
    %echo diffusion tensor imaging of the human heart', or a modified
    %version. This latter version changes the starting position of the
    %first diffusion gradient compared to the original MOCO. To compute the
    %original version set 'modified=0'.
    modified=0;
    
    % Physical constants
    GAMMA = 42.58e3;
    G_Max = G_Max*Gvec;
    S_Max = S_Max*Gvec;
    
    % Simulation constants
    n = length(grad);
    tDiff=n*dt/(1e-3);
    
    ADCcont = ceil(T_ECHO*1e-3/dt)*dt/(1e-3); %EPI time to the center of the k-space
    preTime = ceil(T_90*1e-3/dt)*dt/(1e-3); %end of RF90.
    RFTime = ceil(T_RF*1e-3/dt)*dt/(1e-3);
    
    tECHO = n + ADCcont/(dt*1e3);
    
    % Check times are ok. In this case there is another check point later
    % on. 
    if tECHO/2+T_90*1e-3/dt>n
        fprintf('MOCO: The sequence times are not valid. T_EPI is longer that T_Diff\n');
        G_MOCO=grad;
        b_val=-1;
        phaseFinal=-1;
        return;
    end
    
    tINV = floor(tECHO/2);
    INV = ones(n,1);   INV(tINV:end) = -1;
    
    C=tril(ones(n));
    C2 = C'*C;
    
    D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
    D = D(1:end-1,:)/dt;
    
    t1=floor(preTime/(dt*1e3)); 
    t2=tINV+ceil(RFTime/(dt*1e3)/2); 
    n_after_RF=n-t2;
    Tu=n_after_RF*dt/1e-3; 
    nepsilon=floor(G_Max/S_Max/dt)+2; 
    epsilon=nepsilon*dt/1e-3; 
    lamda=(Tu-5*epsilon)/3; 
    nlamda=ceil(lamda*1e-3/dt);
    
    slopeUP=linspace(0,G_Max,nepsilon); 
    slopeDOWN=linspace(G_Max,0,nepsilon);
    point=t2;
    grad(t2:t2+nepsilon-1)=-slopeUP; %epsilon, gradient ramp time
    point=t2+nepsilon-1;
    grad(point+1:point+1+2*nlamda+nepsilon)=-G_Max;
    point=point+1+2*nlamda+nepsilon; %2*nlamda+nepsilon is the width of the lower lobe.
    grad(point:point+nepsilon-1)=-slopeDOWN;
    point=point+nepsilon-1;
    grad(point:point+nepsilon-1)=slopeUP;
    point=point+nepsilon-1;
    grad(point:end-nepsilon)=G_Max;
    grad(end-nepsilon+1:end)=slopeDOWN;
    
    
    %Check the size of the right lobe does guarantees the G(RF90)=0
    %constraint.
    if (t2-(2*nepsilon+nlamda)-size(grad(t2+1:end),1))<floor(preTime/(dt*1e3))
        fprintf('WARNING! Losing M2 nulling to be able to make the waveform.\n');
        tmp=floor(preTime/(dt*1e3))-(t2-(2*nepsilon+nlamda)-size(grad(t2+1:end),1));
    else %if everything is fine
        tmp=0;
    end
    % original MOCO
    if ~modified
        grad(tmp+t2-(2*nepsilon+nlamda)-size(grad(t2+1:end),1):tmp+t2-(2*nepsilon+nlamda)-1)=flipud(grad(t2+1:end));

        % MODIFIED MOCO so the location of the first lobe does not depend on the 
        % MOCO parameteres. Apparently, finishing the gradient at the
        % beginning of the RF180 gives a better M2 nulling.
    else
        %we set the end of the first gradient lobe at the beginning of the
        %RF180
        grad(t2-n_after_RF-ceil(RFTime/(dt*1e3))+1:t2-ceil(RFTime/(dt*1e3)))=flipud(grad(t2+1:end));

    end
    
    
    b_val = (GAMMA*2*pi)^2*(grad.*INV*dt)'*(C2*(grad.*INV*dt))*dt;
    
    %if we want a target b-value.
    if bvalue_target~=0 && bvalue_target<b_val
        [grad, b_val] = derating(grad,bvalue_target,INV,dt,C2,GAMMA);
    end
    
    % diffusion encoding duration
    tDiff = length(grad)*dt/(1e-3);
    TE = tDiff + ADCcont;
    
    t0 = 0;

    %%Check results
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
    
    % moments and phase over time
    M0 = GAMMA*dt*tMat0*(grad.*INV);
    M1 = GAMMA*dt*tMat1*(grad.*INV);
    M2 = GAMMA*dt*tMat2*(grad.*INV);
    phaseFinal = dt*tMat0*(grad.^2.*INV);
    
    G_MOCO = grad;
    
    DESCRIPTION = ['bValue: ' num2str(round(b_val)) ', TE: ' num2str(TE) '---Gmax: ' num2str(max(G_MOCO)/Gvec)  '---SRmax: ' num2str(max(abs(D*G_MOCO/Gvec)))  ];
    
    %% Save waveform
    grad_opt=G_MOCO;
    
    %% Generate a figure
%     figure; subplot(3,1,1);
%     plot(G_MOCO,'LineWidth',2); hold on; %plot(start);
%     title(DESCRIPTION); ylabel('G waveform');
%     subplot(3,1,2);
%     plot(M1/100,'r','LineWidth',2); hold on; plot(M2,'LineWidth',2);ylabel('Moments');
%     legend('m1','m2','location','northwest');
%     subplot(3,1,3);
%     plot(abs(D*G_MOCO/Gvec),'LineWidth',2);ylabel('dG/dt'); %slow rate

    phaseFinal=phaseFinal(end);
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

