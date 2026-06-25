%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UNIVERSAL SAMPLE / MOF EXTRACTION + CONDUCTIVITY + DRUDE-SMITH  (CORRECTED)
% Geometry: air / substrate / sample / substrate / air
%
% Step 2 of 2: run AFTER Cuvette_code_corrected.m, which writes the substrate
% index files (Substrate_%dK_tr.dat_n_ex_real / _n_ex_imag) read in below.
%
% ---------------------------------------------------------------------------
% CORRECTIONS vs the original sample_code_universal.m
%   1. Speed of light set to the exact SI value (299792458 m/s; was 2.997925e8).
%
% VERIFIED CORRECT, KEPT AS-IS (annotated below so they are not "fixed" by mistake):
%   * Fabry-Perot sign.  Both etalons use (1 + r.*r.*phase.^2); with r34 = -r23
%     this is the textbook 1/(1 - r^2 P^2).  This is the CORRECT sign — the
%     substrate (Cuvette) code is the one that had the slip; it is fixed there.
%   * Locked spacer.  The sample layer and the empty reference gap share the same
%     thickness L3 (phase3 and phase3ref both use L3), so the filled/empty paths
%     stay symmetric and the thin-layer Fabry-Perot is modelled consistently.
%   * Fabry-Perot kept ON.  The thin sample / gap (~0.1 mm => ~0.9 ps round trip)
%     cannot be time-gated, so the internal reflection must be modelled, not
%     dropped.  (The mm-scale substrate windows ARE gated — no window etalon term.)
% ---------------------------------------------------------------------------

clear all; close all; clc;

Temps = [77 100 150 200 250 300];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Search range for sample/MOF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nmin = 1.0;
nmax = 2.5;
nn = 700;

kmin = 0;
kmax = -0.2;
nk = 500;

n_axis = linspace(nmin,nmax,nn);
k_axis = linspace(kmin,kmax,nk);

ns = zeros(nn,nk);

for x = 1:nn
    for y = 1:nk
        ns(x,y) = n_axis(x) + 1i*k_axis(y);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fmin = 0.8e12;
fmax = 2.0e12;
nfreq_target = 200;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thicknesses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L2 = 1e-3;        % first substrate
L3 = 0.13e-3;     % sample/MOF layer  (LOCKED SPACER: also the empty-gap thickness)
L4 = 1e-3;        % second substrate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_air = 1.00055;
NFourier = 2^12;
c = 299792458;    % CORRECTED: exact SI value (was 2.997925e8)
eps0 = 8.8541878128e-12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Drude-Smith fit window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fit_min_THz = 0.8;
fit_max_THz = 2.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AllFreq = cell(length(Temps),1);
AllN = cell(length(Temps),1);
AllEpsReal = cell(length(Temps),1);
AllEpsImag = cell(length(Temps),1);
AllSigma = cell(length(Temps),1);
AllSigmaFit = cell(length(Temps),1);

DS_omega_p = zeros(length(Temps),1);
DS_tau = zeros(length(Temps),1);
DS_c = zeros(length(Temps),1);
DS_sigmaDC = zeros(length(Temps),1);
DS_resnorm = zeros(length(Temps),1);
% Part 2 — start temperature loop and load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temperature loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for tt = 1:length(Temps)

    T = Temps(tt);

    FileEref = sprintf('Substrate_%dK_tr.dat',T);
    FileEtrans = sprintf('Sample_%dK_tr.dat',T);

    fprintf('\n========================================\n');
    fprintf('Processing sample at %d K\n',T);
    fprintf('Reference: %s\n',FileEref);
    fprintf('Sample:    %s\n',FileEtrans);
    fprintf('========================================\n');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load substrate refractive index from substrate extraction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    glass_real = dlmread(sprintf('%s%s', FileEref, '_n_ex_real'));
    glass_imag = dlmread(sprintf('%s%s', FileEref, '_n_ex_imag'));

    n2f = glass_real + 1i*glass_imag;
    n4f = glass_real + 1i*glass_imag;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load time-domain data: two columns [time, field]
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Eref_raw = dlmread(FileEref);
    Etrans_raw = dlmread(FileEtrans);

    time_ref = Eref_raw(:,1);

    Eref = Eref_raw(:,2);
    Etrans = Etrans_raw(:,2);

    % Time axis assumed in ps
    Timestep = mean(diff(time_ref))*1e-12;

    % Remove offset
    Eref = Eref - mean(Eref(1:5));
    Etrans = Etrans - mean(Etrans(1:5));

    % Symmetric padding (identical on both traces -> no spurious relative delay)
    Eref = [zeros(150,1); Eref; zeros(150,1)];
    Etrans = [zeros(150,1); Etrans; zeros(150,1)];
% Part 3 — windowing and FFT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Window reference pulse
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [~, PeakTimeRef] = max(abs(Eref));

    NHam = round(4e-12/Timestep);
    Ham = hamming(NHam);

    WindowRef = zeros(size(Eref));

    start_ref = PeakTimeRef - ceil(NHam/2);
    stop_ref = start_ref + NHam - 1;

    start_ref = max(1,start_ref);
    stop_ref = min(length(Eref),stop_ref);

    WindowRef(start_ref:stop_ref) = Ham(1:(stop_ref-start_ref+1));
    Eref = Eref .* WindowRef;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Window sample pulse
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [~, PeakTimeTrans] = max(abs(Etrans));

    WindowTrans = zeros(size(Etrans));

    start_trans = PeakTimeTrans - ceil(NHam/2);
    stop_trans = start_trans + NHam - 1;

    start_trans = max(1,start_trans);
    stop_trans = min(length(Etrans),stop_trans);

    WindowTrans(start_trans:stop_trans) = Ham(1:(stop_trans-start_trans+1));
    Etrans = Etrans .* WindowTrans;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FFT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Eref = fft(Eref,NFourier);
    Etrans = fft(Etrans,NFourier);

    freq_full = (1/Timestep)*(0:(NFourier-1))/NFourier;
    freqstep = 1/Timestep/NFourier;

    Nfmin = floor(fmin/freqstep);
    Nfmax = floor(fmax/freqstep);

    Nfmin = max(1,Nfmin);

    Nfreqstep = max(1,floor((Nfmax-Nfmin)/nfreq_target));

    freq = freq_full(Nfmin:Nfreqstep:Nfmax);

    Eref = Eref(Nfmin:Nfreqstep:Nfmax);
    Etrans = Etrans(Nfmin:Nfreqstep:Nfmax);

    DeltaE = Etrans - Eref;

    nfreq = length(freq)-1;

    if length(n2f) ~= length(freq)
        warning('Substrate n length does not match current frequency grid at %d K.',T);
        warning('Rerun substrate extraction using the same frequency settings.');
    end

    RefrIndex = zeros(1,nfreq+1);
    Minima = zeros(nfreq+1,nn);
% Part 4 — sample extraction loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Frequency loop: extract sample n
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for F = 1:nfreq+1

        n1 = n_air .* ones(nn,nk);
        n2 = n2f(F) .* ones(nn,nk);
        n3 = ns;
        n4 = n4f(F) .* ones(nn,nk);
        n5 = n_air .* ones(nn,nk);
        nref = n_air .* ones(nn,nk);

        t23 = 2*n2 ./ (n2+n3);
        t34 = 2*n3 ./ (n3+n4);

        t23ref = 2*n2 ./ (n2+nref);
        t34ref = 2*nref ./ (nref+n4);

        r23 = (n3-n2) ./ (n2+n3);
        r34 = (n4-n3) ./ (n3+n4);

        r23ref = (nref-n2) ./ (n2+nref);
        r34ref = (n4-nref) ./ (nref+n4);

        Ratio_meas = ones(nn,nk) .* (DeltaE(F)./Eref(F));

        % LOCKED SPACER: sample layer and empty gap share L3.
        phase3 = exp(-1i*n3.*2*pi*freq(F)*L3./c);
        phase3ref = exp(-1i*nref.*2*pi*freq(F)*L3./c);

        % Fabry-Perot ON, correct sign: (1 + r.*r.*phase.^2) = 1/(1 - r^2 P^2)
        % since r34 = -r23.  Sample etalon on the sample layer, reference etalon
        % on the empty air gap; both non-gatable, so both are modelled.
        Ratio_calc = ...
            ((phase3 .* t23 .* t34 ./ (1 + r23 .* r34 .* phase3.^2)) ./ ...
            ((phase3ref .* t23ref .* t34ref) ./ ...
            (1 + r23ref .* r34ref .* phase3ref.^2))) - 1;

        Diff = abs(real(Ratio_calc - Ratio_meas)) + ...
               abs(imag(Ratio_calc - Ratio_meas));

        % Branch-tracking minimization
        if F == 1

            [~, idx] = min(Diff(:));
            [MinRow, MinColumn] = ind2sub(size(Diff),idx);

        else

            previous_n = real(RefrIndex(F-1));
            previous_k = imag(RefrIndex(F-1));

            [~, center_row] = min(abs(n_axis - previous_n));
            [~, center_col] = min(abs(k_axis - previous_k));

            row_window = max(1,center_row-15):min(nn,center_row+15);
            col_window = max(1,center_col-15):min(nk,center_col+15);

            Diff_local = Diff(row_window,col_window);

            [~, idx] = min(Diff_local(:));
            [local_row, local_col] = ind2sub(size(Diff_local),idx);

            MinRow = row_window(local_row);
            MinColumn = col_window(local_col);

        end

        RefrIndex(F) = ns(MinRow,MinColumn);
        Minima(F,:) = Diff(:,MinColumn);

    end
% Part 5 — smoothing, dielectric function, conductivity
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Smooth n mildly
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    RefrIndex_raw = RefrIndex;

    RefrIndex = smooth(real(RefrIndex),5).' + ...
                1i*smooth(imag(RefrIndex),5).';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dielectric function
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eps_complex = RefrIndex.^2;

    eps_real = real(eps_complex);
    eps_imag = -imag(eps_complex);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Real conductivity
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    omega = 2*pi*freq;

    sigma_real = omega .* eps0 .* eps_imag;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Store spectra
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    AllFreq{tt} = freq;
    AllN{tt} = RefrIndex;
    AllEpsReal{tt} = eps_real;
    AllEpsImag{tt} = eps_imag;
    AllSigma{tt} = sigma_real;
% Part 6 — Drude–Smith fit
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Drude-Smith fit to real conductivity
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    sigma1 = sigma_real(:);

    f_Hz = freq(:);
    f_THz = freq(:)/1e12;

    good = isfinite(f_Hz) & isfinite(f_THz) & isfinite(sigma1);

    f_Hz = f_Hz(good);
    f_THz = f_THz(good);
    sigma1 = sigma1(good);

    [f_Hz, order] = sort(f_Hz);
    f_THz = f_THz(order);
    sigma1 = sigma1(order);

    idx = find(f_THz >= fit_min_THz & f_THz <= fit_max_THz);

    if isempty(idx)
        warning('No frequency points in DS fitting window for %d K.',T);
        continue
    end

    f_fit_Hz = f_Hz(idx);
    y_fit = sigma1(idx);

    omega_fit = 2*pi*f_fit_Hz;

    sigma1_DS = @(q) real( ...
        eps0*(10.^q(1)).^2 .* (10.^q(2)) .* ...
        (1 ./ (1 - 1i*omega_fit*(10.^q(2)))) .* ...
        (1 + q(3) ./ (1 - 1i*omega_fit*(10.^q(2)))) );

    scale_y = max(abs(y_fit));

    if scale_y == 0
        warning('Conductivity is zero in DS fitting window for %d K.',T);
        continue
    end

    residual = @(q) (sigma1_DS(q) - y_fit)./scale_y;

    q0 = [14, log10(100e-15), -0.5];

    lb = [10, -15, -0.99];
    ub = [17, -12,  0];

    opts = optimoptions('lsqnonlin', ...
        'Display','off', ...
        'FiniteDifferenceType','central', ...
        'MaxIterations',5000, ...
        'MaxFunctionEvaluations',5e4);

    [q_fit, resnorm] = lsqnonlin(residual,q0,lb,ub,opts);

    omega_p = 10.^q_fit(1);
    tau = 10.^q_fit(2);
    c_DS = q_fit(3);

    sigma_dc = eps0*omega_p^2*tau*(1+c_DS);

    DS_omega_p(tt) = omega_p;
    DS_tau(tt) = tau;
    DS_c(tt) = c_DS;
    DS_sigmaDC(tt) = sigma_dc;
    DS_resnorm(tt) = resnorm;

    omega_full = 2*pi*f_Hz;

    sigma1_fit_full = real( ...
        eps0*omega_p^2*tau .* ...
        (1 ./ (1 - 1i*omega_full*tau)) .* ...
        (1 + c_DS ./ (1 - 1i*omega_full*tau)) );

    AllSigmaFit{tt} = sigma1_fit_full;

    fprintf('\n=== %d K DRUDE-SMITH FIT ===\n',T);
    fprintf('omega_p = %.6e rad/s\n',omega_p);
    fprintf('tau     = %.6e s  %.2f fs\n',tau,tau*1e15);
    fprintf('c       = %.6f\n',c_DS);
    fprintf('sigma_dc = %.6e S/m\n',sigma_dc);
    fprintf('resnorm = %.6e\n',resnorm);
% Part 7 — save temperature results and close loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save results for this temperature
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    outbase = sprintf('Sample_%dK_tr.dat_MOF',T);

    fid = fopen(sprintf('%s_real',outbase),'wt');
    fprintf(fid,'%12.8f\n',real(RefrIndex));
    fclose(fid);

    fid = fopen(sprintf('%s_imag',outbase),'wt');
    fprintf(fid,'%12.8f\n',imag(RefrIndex));
    fclose(fid);

    fid = fopen(sprintf('%s_freq',outbase),'wt');
    fprintf(fid,'%12.8f\n',freq);
    fclose(fid);

    fid = fopen(sprintf('%s_eps_real',outbase),'wt');
    fprintf(fid,'%12.8f\n',eps_real);
    fclose(fid);

    fid = fopen(sprintf('%s_eps_imag',outbase),'wt');
    fprintf(fid,'%12.8f\n',eps_imag);
    fclose(fid);

    fid = fopen(sprintf('%s_sigma_real',outbase),'wt');
    fprintf(fid,'%12.8f\n',sigma_real);
    fclose(fid);

    fprintf('Finished sample extraction at %d K\n',T);

end
% Part 8 — combined spectral plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combined plots: refractive index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt})
        plot(AllFreq{tt}/1e12, real(AllN{tt}), 'o-', ...
            'DisplayName',sprintf('%d K',Temps(tt)));
    end
end

xlabel('Frequency (THz)');
ylabel('n');
title('Sample real refractive index');
legend('Location','best');
grid on;

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt})
        plot(AllFreq{tt}/1e12, -imag(AllN{tt}), 'o-', ...
            'DisplayName',sprintf('%d K',Temps(tt)));
    end
end

xlabel('Frequency (THz)');
ylabel('k');
title('Sample imaginary refractive index');
legend('Location','best');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combined plots: permittivity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt})
        plot(AllFreq{tt}/1e12, AllEpsReal{tt}, 'o-', ...
            'DisplayName',sprintf('%d K',Temps(tt)));
    end
end

xlabel('Frequency (THz)');
ylabel('\epsilon''');
title('Sample real permittivity');
legend('Location','best');
grid on;

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt})
        plot(AllFreq{tt}/1e12, AllEpsImag{tt}, 'o-', ...
            'DisplayName',sprintf('%d K',Temps(tt)));
    end
end

xlabel('Frequency (THz)');
ylabel('\epsilon''''');
title('Sample imaginary permittivity');
legend('Location','best');
grid on;
% Part 9 — conductivity and Drude–Smith plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combined conductivity plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt})
        plot(AllFreq{tt}/1e12, AllSigma{tt}, 'o-', ...
            'DisplayName',sprintf('%d K',Temps(tt)));
    end
end

xlabel('Frequency (THz)');
ylabel('\sigma_1 (S/m)');
title('Sample real conductivity');
legend('Location','best');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Drude-Smith fits over conductivity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on;

for tt = 1:length(Temps)
    if ~isempty(AllFreq{tt}) && ~isempty(AllSigmaFit{tt})
        plot(AllFreq{tt}/1e12, AllSigma{tt}, 'o', ...
            'DisplayName',sprintf('%d K data',Temps(tt)));

        plot(AllFreq{tt}/1e12, AllSigmaFit{tt}, '-', ...
            'DisplayName',sprintf('%d K fit',Temps(tt)));
    end
end

xline(fit_min_THz,'--');
xline(fit_max_THz,'--');

xlabel('Frequency (THz)');
ylabel('\sigma_1 (S/m)');
title('Drude-Smith fits');
legend('Location','best');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Drude-Smith parameters vs temperature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(Temps,DS_tau*1e15,'o-');
xlabel('Temperature (K)');
ylabel('\tau (fs)');
title('Drude-Smith scattering time');
grid on;

figure;
plot(Temps,DS_c,'o-');
xlabel('Temperature (K)');
ylabel('c');
title('Drude-Smith localization parameter');
grid on;

figure;
plot(Temps,DS_omega_p,'o-');
xlabel('Temperature (K)');
ylabel('\omega_p (rad/s)');
title('Drude-Smith plasma frequency');
grid on;

figure;
plot(Temps,DS_sigmaDC,'o-');
xlabel('Temperature (K)');
ylabel('\sigma_{dc} (S/m)');
title('Drude-Smith DC conductivity');
grid on;

figure;
plot(Temps,DS_resnorm,'o-');
xlabel('Temperature (K)');
ylabel('resnorm');
title('Drude-Smith fit residual norm');
grid on;
