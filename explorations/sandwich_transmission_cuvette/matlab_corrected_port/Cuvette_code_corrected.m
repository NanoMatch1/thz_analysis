%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UNIVERSAL SUBSTRATE EXTRACTION  (CORRECTED RE-PORT)
% Geometry: air / substrate / air gap / substrate / air
%
% Step 1 of 2: extract the substrate (window) refractive index from the
% empty-cuvette measurement, to feed sample_code_corrected.m (step 2).
%
% ---------------------------------------------------------------------------
% CORRECTIONS vs the original Cuvette_code_universal.m  (all marked "CORRECTED")
%   1. Fabry-Perot sign-slip FIXED.  The gap etalon was written
%        (1 - r23.*r34.*phase3.^2)
%      but with r34 = -r23 that evaluates to (1 + r^2 P^2) — the wrong sign.
%      The textbook factor is 1/(1 - r^2 P^2); with this code's r-convention
%      that is written (1 + r23.*r34.*phase3.^2).  This now matches the sample
%      code, which always had the correct sign.
%   2. Zero-padding made SYMMETRIC.  The original padded the reference and the
%      transmitted pulse by DIFFERENT amounts (150/210 vs 210/150) — a 60-sample
%      (~3 ps) RELATIVE shift between them.  That spurious group delay inflated
%      the extracted index (fused silica read ~2.2-2.5 instead of ~1.95).
%      Identical padding on both traces preserves the true inter-pulse delay.
%   3. Search range widened.  The original 1.71-1.80 rails BELOW the true
%      fused-silica value (~1.95) once the padding is fixed.  Widened to cover it.
%   4. Speed of light set to the exact SI value (299792458 m/s).
%   5. Leading-baseline (DC offset) removal added, matching the sample code.
% ---------------------------------------------------------------------------

clear all; close all; clc;

Temps = [77 100 150 200 250 300];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Substrate search range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nmin = 1.70;      % CORRECTED: was 1.71 (railed below true ~1.95 after padding fix)
nmax = 2.30;      % CORRECTED: was 1.80
nn = 500;

kmin = 0;
kmax = -0.03;
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

L2 = 1e-3;        % first substrate window  (set to your measured window thickness)
L3 = 0.12e-3;     % air gap between the two windows
L4 = 1e-3;        % second substrate window

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_air = 1.00055;
NFourier = 2^12;
c = 299792458;    % CORRECTED: exact SI value (was 2.997925e8)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Storage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AllFreq = cell(length(Temps),1);
AllN = cell(length(Temps),1);

% Part 2: main loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temperature loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for tt = 1:length(Temps)

    T = Temps(tt);

    FileEref = sprintf('Air_%dK_tr.dat',T);
    FileEtrans = sprintf('Substrate_%dK_tr.dat',T);

    fprintf('\nProcessing substrate at %d K\n',T);

    % Load data
    Eref_raw = dlmread(FileEref);
    Etrans_raw = dlmread(FileEtrans);

    time_ref = Eref_raw(:,1);

    Eref = Eref_raw(:,2);
    Etrans = Etrans_raw(:,2);

    Timestep = mean(diff(time_ref))*1e-12;   % time column in ps

    % CORRECTED: remove leading-baseline DC offset before padding (matches sample code)
    Eref = Eref - mean(Eref(1:5));
    Etrans = Etrans - mean(Etrans(1:5));

    % CORRECTED: SYMMETRIC padding — identical on both traces so no spurious
    % relative group delay is introduced (the original used 150/210 vs 210/150).
    PadFront = 180;
    PadBack = 180;
    Eref = [zeros(PadFront,1); Eref; zeros(PadBack,1)];
    Etrans = [zeros(PadFront,1); Etrans; zeros(PadBack,1)];

    % Window reference
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

    % Window substrate
    [~, PeakTimeTrans] = max(abs(Etrans));

    WindowTrans = zeros(size(Etrans));

    start_trans = PeakTimeTrans - ceil(NHam/2);
    stop_trans = start_trans + NHam - 1;

    start_trans = max(1,start_trans);
    stop_trans = min(length(Etrans),stop_trans);

    WindowTrans(start_trans:stop_trans) = Ham(1:(stop_trans-start_trans+1));
    Etrans = Etrans .* WindowTrans;

    % FFT
    DeltaE = Etrans - Eref;

    DeltaE = fft(DeltaE,NFourier);
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
    DeltaE = DeltaE(Nfmin:Nfreqstep:Nfmax);

    nfreq = length(freq)-1;

    RefrIndex = zeros(1,nfreq+1);

% Part 3: fitting loop + saving

    for F = 1:nfreq+1

        n1 = n_air .* ones(nn,nk);
        n2 = ns;
        n3 = n_air .* ones(nn,nk);
        n4 = ns;
        n5 = n_air .* ones(nn,nk);

        t12 = 2*n1 ./ (n1+n2);
        t23 = 2*n2 ./ (n2+n3);
        t34 = 2*n3 ./ (n3+n4);
        t45 = 2*n4 ./ (n4+n5);

        r23 = (n3-n2) ./ (n2+n3);
        r34 = (n4-n3) ./ (n3+n4);

        Ratio_meas = ones(nn,nk) .* (DeltaE(F)./Eref(F));

        phase2 = exp(-1i*n2.*2*pi*freq(F)*L2./c);
        phase3 = exp(-1i*n3.*2*pi*freq(F)*L3./c);
        phase4 = exp(-1i*n4.*2*pi*freq(F)*L4./c);

        phase_ref = exp(-1i*n_air.*2*pi*freq(F)*(L2+L3+L4)./c);

        % CORRECTED: gap Fabry-Perot sign.  Textbook 1/(1 - r^2 P^2); with
        % r34 = -r23 that is written (1 + r23.*r34.*phase3.^2).  The original
        % had (1 - r23.*r34.*phase3.^2), i.e. the wrong sign.
        Ratio_calc = ...
            (phase2.*phase3.*phase4.*t12.*t23.*t34.*t45) ./ ...
            ((1 + r23.*r34.*phase3.^2).*phase_ref) - 1;

        Diff = abs(real(Ratio_calc - Ratio_meas)) + ...
               abs(imag(Ratio_calc - Ratio_meas));

        if F == 1

            [~, idx] = min(Diff(:));
            [MinRow, MinColumn] = ind2sub(size(Diff),idx);

        else

            previous_n = real(RefrIndex(F-1));
            previous_k = imag(RefrIndex(F-1));

            [~, center_row] = min(abs(n_axis - previous_n));
            [~, center_col] = min(abs(k_axis - previous_k));

            row_window = max(1,center_row-20):min(nn,center_row+20);
            col_window = max(1,center_col-20):min(nk,center_col+20);

            Diff_local = Diff(row_window,col_window);

            [~, idx] = min(Diff_local(:));
            [local_row, local_col] = ind2sub(size(Diff_local),idx);

            MinRow = row_window(local_row);
            MinColumn = col_window(local_col);

        end

        RefrIndex(F) = ns(MinRow,MinColumn);

    end

    % Store
    AllFreq{tt} = freq;
    AllN{tt} = RefrIndex;

    % Save
    outbase = sprintf('Substrate_%dK_tr.dat',T);

    fid = fopen(sprintf('%s_n_ex_real',outbase),'wt');
    fprintf(fid,'%12.8f\n',real(RefrIndex));
    fclose(fid);

    fid = fopen(sprintf('%s_n_ex_imag',outbase),'wt');
    fprintf(fid,'%12.8f\n',imag(RefrIndex));
    fclose(fid);

    fid = fopen(sprintf('%s_freq',outbase),'wt');
    fprintf(fid,'%12.8f\n',freq);
    fclose(fid);

    fprintf('Finished %d K\n',T);

end

% Part 4: combined plots

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combined substrate plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
hold on;

for tt = 1:length(Temps)
    plot(AllFreq{tt}/1e12, real(AllN{tt}), 'o-', ...
        'DisplayName',sprintf('%d K',Temps(tt)));
end

xlabel('Frequency (THz)');
ylabel('n');
title('Substrate n');
legend('Location','best');
grid on;

figure;
hold on;

for tt = 1:length(Temps)
    plot(AllFreq{tt}/1e12, -imag(AllN{tt}), 'o-', ...
        'DisplayName',sprintf('%d K',Temps(tt)));
end

xlabel('Frequency (THz)');
ylabel('k');
title('Substrate k');
legend('Location','best');
grid on;

figure;
hold on;

for tt = 1:length(Temps)
    eps_sub = AllN{tt}.^2;
    plot(AllFreq{tt}/1e12, real(eps_sub), 'o-', ...
        'DisplayName',sprintf('%d K',Temps(tt)));
end

xlabel('Frequency (THz)');
ylabel('\epsilon''');
title('Substrate real permittivity');
legend('Location','best');
grid on;

figure;
hold on;

for tt = 1:length(Temps)
    eps_sub = AllN{tt}.^2;
    plot(AllFreq{tt}/1e12, -imag(eps_sub), 'o-', ...
        'DisplayName',sprintf('%d K',Temps(tt)));
end

xlabel('Frequency (THz)');
ylabel('\epsilon''''');
title('Substrate imaginary permittivity');
legend('Location','best');
grid on;
