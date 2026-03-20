"""
Извлечение признаков из URL для классификации
"""
import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Set

# Список сервисов сокращения ссылок
SHORTENERS: Set[str] = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 'is.gd', 'buff.ly',
    'short.link', 't.co', 'fb.me', 'bitly.com', 'tiny.cc', 'tr.im',
    'shorturl.at', 'rb.gy', 'cutt.ly', 'shorte.st', 'v.gd', 'clck.ru',
    's.id', 'gg.gg', 'soo.gd', 'x.co', 'lnkd.in', 'mcaf.ee'
}

# Подозрительные ключевые слова
SUSPICIOUS_KEYWORDS: Set[str] = {
    'secure', 'login', 'verify', 'account', 'update', 'confirm',
    'banking', 'signin', 'signin', 'webscr', 'paypal', 'bank',
    'auth', 'authenticate', 'validate', 'verification', 'unlock',
    'suspend', 'security', 'warning', 'alert', 'notice', 'statement'
}


class URLFeatureExtractor:
    """Извлекает признаки из URL для ML модели"""
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Возвращает список названий признаков"""
        return [
            'url_length',
            'num_dots',
            'num_hyphens',
            'num_underscores',
            'num_slashes',
            'num_equals',
            'num_question_marks',
            'num_and',
            'num_digits',
            'num_letters',
            'has_ip',
            'is_shortened',
            'num_subdomains',
            'domain_length',
            'path_length',
            'query_length',
            'num_suspicious_keywords',
            'has_at_symbol',
            'has_double_slash',
            'entropy',
            'num_redirects_in_path',
            'has_https',
            'has_http_only',
            'percent_encoded_chars',
            'path_depth'
        ]
    
    def extract(self, url: str) -> Dict[str, float]:
        """
        Извлекает все признаки из URL
        
        Args:
            url: строка URL
            
        Returns:
            словарь с признаками
        """
        features = {}
        
        try:
            parsed = urlparse(url if '://' in url else 'http://' + url)
        except:
            # Если парсинг не удался, возвращаем значения по умолчанию
            return {name: 0.0 for name in self.feature_names}
        
        # Базовые признаки
        features['url_length'] = len(url)
        
        # Количество специальных символов
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_equals'] = url.count('=')
        features['num_question_marks'] = url.count('?')
        features['num_and'] = url.count('&')
        
        # Количество цифр и букв
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        # Проверка на IP-адрес вместо домена
        features['has_ip'] = 1.0 if self._contains_ip(parsed.netloc) else 0.0
        
        # Проверка на сокращатель ссылок
        domain = parsed.netloc.lower()
        features['is_shortened'] = 1.0 if any(short in domain for short in SHORTENERS) else 0.0
        
        # Количество поддоменов
        domain_parts = domain.split('.')
        features['num_subdomains'] = max(0, len(domain_parts) - 2) if len(domain_parts) > 2 else 0.0
        
        # Длина домена
        features['domain_length'] = len(domain)
        
        # Длина пути
        features['path_length'] = len(parsed.path)
        
        # Длина query параметров
        features['query_length'] = len(parsed.query)
        
        # Количество подозрительных ключевых слов в URL
        url_lower = url.lower()
        features['num_suspicious_keywords'] = sum(
            1 for keyword in SUSPICIOUS_KEYWORDS if keyword in url_lower
        )
        
        # Наличие символа @
        features['has_at_symbol'] = 1.0 if '@' in url else 0.0
        
        # Наличие двойного слеша
        features['has_double_slash'] = 1.0 if '//' in parsed.netloc else 0.0
        
        # Энтропия URL (чем выше, тем более случайный/подозрительный)
        features['entropy'] = self._calculate_entropy(url)
        
        # Количество редиректов в пути (признак подозрительности)
        features['num_redirects_in_path'] = parsed.path.count('/redirect') + parsed.path.count('/go/')
        
        # Использование HTTPS
        features['has_https'] = 1.0 if parsed.scheme == 'https' else 0.0
        features['has_http_only'] = 1.0 if parsed.scheme == 'http' else 0.0
        
        # Процент закодированных символов (%XX)
        encoded_chars = re.findall(r'%[0-9A-Fa-f]{2}', url)
        features['percent_encoded_chars'] = len(encoded_chars) / max(len(url), 1)
        
        # Глубина пути (количество сегментов)
        path_segments = [s for s in parsed.path.split('/') if s]
        features['path_depth'] = len(path_segments)
        
        # Нормализация некоторых признаков
        features['url_length'] = min(features['url_length'] / 2000, 1.0)
        features['domain_length'] = min(features['domain_length'] / 100, 1.0)
        features['path_length'] = min(features['path_length'] / 500, 1.0)
        features['num_subdomains'] = min(features['num_subdomains'] / 5, 1.0)
        features['num_suspicious_keywords'] = min(features['num_suspicious_keywords'] / 10, 1.0)
        
        return features
    
    def extract_as_array(self, url: str) -> list:
        """Извлекает признаки в виде массива для ML модели"""
        features = self.extract(url)
        return [features[name] for name in self.feature_names]
    
    def _contains_ip(self, hostname: str) -> bool:
        """Проверяет, является ли hostname IP-адресом"""
        # Простая проверка на IPv4
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, hostname):
            parts = hostname.split('.')
            return all(0 <= int(p) <= 255 for p in parts)
        
        # Проверка на IPv6
        if ':' in hostname:
            return True
        
        return False
    
    def _calculate_entropy(self, s: str) -> float:
        """Вычисляет энтропию Шеннона для строки"""
        if not s:
            return 0.0
        
        prob = [float(s.count(c)) / len(s) for c in set(s)]
        entropy = -sum(p * math.log2(p) for p in prob)
        
        # Нормализуем до [0, 1]
        max_entropy = math.log2(min(len(set(s)), len(s)))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return min(entropy, 1.0)
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список признаков"""
        return self.feature_names.copy()


# Для обратной совместимости
def extract_features(url: str) -> Dict[str, float]:
    """Функция для быстрого извлечения признаков"""
    extractor = URLFeatureExtractor()
    return extractor.extract(url)
