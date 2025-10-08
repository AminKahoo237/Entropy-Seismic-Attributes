
function H = shannon_entropy_2d(signal)
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی دقیق
    signal = signal(:);  % تبدیل به بردار
    [counts, ~] = histcounts(signal, 'Normalization', 'probability');
    P = counts(counts > 0);  % حذف جعبه‌های با احتمال صفر
    H = -sum(P .* log2(P));  % محاسبه آنتروپی شانون
end


function H = katz_entropy_2d(signal)
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی
    signal = signal(:);  % تبدیل به بردار
    N = length(signal);
    L = sum(abs(diff(signal)));  % تغییرات کل
    A = max(signal) - min(signal);  % دامنه
    H = log(N) / log(1 + (L / A));  % آنتروپی کاتسای
end

% آنتروپی تیکنی (Tsallis Entropy) برای داده‌های دو‌بعدی
function H = tsallis_entropy_2d(signal, q)
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی
    signal = signal(:);  % تبدیل به بردار
    [counts, ~] = histcounts(signal, 'Normalization', 'probability');
    P = counts(counts > 0);  % حذف جعبه‌های با احتمال صفر
    H = (1 - sum(P.^q)) / (q - 1);  % محاسبه آنتروپی تیکنی
end

% آنتروپی فراکتالی (Fractal Entropy) برای داده‌های دو‌بعدی
function H = fractal_entropy_2d(signal)
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی
    signal = signal(:);  % تبدیل به بردار
    N = length(signal);
    kmax = floor(N / 2);  % حداکثر مقدار برای k
    H = 0;  % مقدار اولیه آنتروپی
    for k = 1:kmax
        L = zeros(k, 1);
        for m = 1:k
            L(m) = sum(abs(diff(signal(m:m + k - 1))));
        end
        H = H + log(sum(L) / k) / log(k);  % تخمین ابعاد فراکتالی
    end
end

% آنتروپی رنی برای داده‌های دو‌بعدی
function H = renyi_entropy_2d(signal, alpha)
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی
    signal = signal(:);  % تبدیل به بردار
    [counts, ~] = histcounts(signal, 'Normalization', 'probability');
    P = counts(counts > 0);  % حذف جعبه‌های با احتمال صفر
    H = (1 / (1 - alpha)) * log(sum(P .^ alpha));  % محاسبه آنتروپی رنی
end

% آنتروپی Lempel-Ziv برای داده‌های دو‌بعدی
function H = lempel_ziv_entropy_2d(signal)
%     نرمال‌سازی سیگنال دو‌بعدی
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));  % نرمال‌سازی
    signal = signal(:);  % تبدیل به بردار
    
%     مدیریت NaN و Inf
    signal(isnan(signal)) = 0;  % مقادیر NaN را به صفر تبدیل می‌کنیم
    signal(isinf(signal)) = 0;  % مقادیر Inf را به صفر تبدیل می‌کنیم
    
%     مقیاس‌بندی به محدوده [0, 255]
    signal = round(signal * 255);  % مقیاس‌بندی به محدوده [0, 255]
    
%     اطمینان از اینکه تمام مقادیر در محدوده صحیح قرار دارند
    signal(signal < 0) = 0;  % مقادیر منفی را به 0 تبدیل می‌کنیم
    signal(signal > 255) = 255;  % مقادیر بزرگتر از 255 را به 255 تبدیل می‌کنیم
    
%     تبدیل سیگنال به رشته باینری
    binary_signal = dec2bin(signal, 8);  % تبدیل به باینری با طول 8 بیت
    
%     محاسبه آنتروپی Lempel-Ziv
    n = length(binary_signal);
    L = 1;
    Z = {};
    for i = 1:n
        word = binary_signal(i, :);
        if ~ismember(word, Z)
            Z{end + 1} = word;  % افزودن واژه جدید به لیست
        else
            L = L + 1;
        end
    end
    H = -log(L) / log(n);  % محاسبه آنتروپی Lempel-Ziv
end
% آنتروپی Espinosa برای داده‌های دو‌بعدی
function H = bidimensional_espinosa_entropy(signal)
    % نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
    % استفاده از یک الگوریتم خاص برای محاسبه Espinosa Entropy
    % در اینجا از یک تابع ساده برای محاسبه آنتروپی استفاده می‌کنیم
    esp_value = mean(abs(diff(signal)));
    
    % محاسبه آنتروپی Espinosa
    H = -log(esp_value);
end
% آنتروپی Permutation برای داده‌های دو‌بعدی
function H = bidimensional_permutation_entropy(signal, m, r)
%     نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
%     اندازه سیگنال
    N = length(signal);
    
%     ساختار توزیع احتمال برای جایگشت‌ها
    count = zeros(factorial(m), 1); % شمارش هر نوع جایگشت
    all_perms = perms(1:m); % تمام جایگشت‌ها برای یک بردار از طول m

%     ایجاد یک آرایه برای تمام بخش‌های بردار
    for i = 1:N - m + 1
        sub_signal = signal(i:i + m - 1); % بخش m-بعدی
        
%         مرتب‌سازی مقادیر و تبدیل به ترتیب
        [~, idx] = sort(sub_signal);
        
%         شبیه‌سازی ترتیب‌ها با جایگشت‌های موجود
        [~, perm_idx] = ismember(idx', all_perms, 'rows');
        
%         افزایش شمارش جایگشت
        if perm_idx > 0
            count(perm_idx) = count(perm_idx) + 1;
        end
    end
    
%     محاسبه توزیع احتمال
    prob = count / sum(count);
    
%     محاسبه آنتروپی
    H = -sum(prob .* log2(prob + eps));
end


% آنتروپی انحراف برای داده‌های دو‌بعدی
function H = bidimensional_dispersion_entropy(signal)
%     نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
%     محاسبه انحراف معیار (Dispersion)
    disp_value = std(signal);
    
%     محاسبه آنتروپی انحراف
    H = -log(disp_value);
end
% آنتروپی توزیع برای داده‌های دو‌بعدی
function H = bidimensional_distribution_entropy(signal)
%     نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
%     محاسبه هیستوگرام و توزیع احتمال
    [counts, ~] = histcounts(signal, 'Normalization', 'probability');
    P = counts(counts > 0);  % حذف جعبه‌های با احتمال صفر
    
%     محاسبه آنتروپی توزیع
    H = -sum(P .* log(P));
end
% آنتروپی فازی برای داده‌های دو‌بعدی
function H = bidimensional_fuzzy_entropy(signal)
    % نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
    % استفاده از تابع فازی برای محاسبه آنتروپی
    fuzz_value = mean(abs(signal - mean(signal)));
    
    % محاسبه آنتروپی فازی
    H = -log(fuzz_value);
end
% آنتروپی Samp برای داده‌های دو‌بعدی
function H = bidimensional_samp_entropy(signal, m, r)
    % نرمال‌سازی سیگنال
    signal = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
    signal = signal(:);
    
    % اندازه سیگنال
    N = length(signal);
    
    % ساختن بردارهای m-بعدی
    phi_m = zeros(N - m, 1);
    phi_m1 = zeros(N - m - 1, 1);
    
    % این حلقه برای محاسبه phi_m (آنتروپی Samp برای m)
    for i = 1:N - m
        xi = signal(i:i + m - 1);  % بخش از سیگنال با طول m
        
        % مقایسه بخش‌های سیگنال
        diff = abs(repmat(xi, N - m, 1) - signal(1:N - m)');  % اصلاح ابعاد با استفاده از ' برای تبدیل به ردیف
        phi_m(i) = sum(all(diff <= r, 2)) / (N - m);  % مقایسه کل بخش‌ها
    end
    
    % این حلقه برای محاسبه phi_m1 (آنتروپی Samp برای m+1)
    for i = 1:N - m - 1
        xi = signal(i:i + m);  % بخش از سیگنال با طول m+1
        
        % مقایسه بخش‌های سیگنال
        diff = abs(repmat(xi, N - m - 1, 1) - signal(1:N - m - 1)');  % اصلاح ابعاد با استفاده از ' برای تبدیل به ردیف
        phi_m1(i) = sum(all(diff <= r, 2)) / (N - m - 1);  % مقایسه کل بخش‌ها
    end
    
    % محاسبه آنتروپی Samp
    H = -log(sum(phi_m1) / sum(phi_m));
end


