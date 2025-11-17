# AI-suggested implementation (built-in sort; stable, C optimized)
def sort_dicts_ai(arr, key_name, default=None, reverse=False):
    """
    Sorts list of dicts by key_name using built-in sort.
    Handles missing keys by using dict.get(key_name, default).
    """
    return sorted(arr, key=lambda d: d.get(key_name, default), reverse=reverse)


# filepath: c:\Users\Kioko\Desktop\PLP ACADEMY\AI SESSION\WK 4\ASSIGNMENT\task1_sort.py
# Manual implementation (merge sort to guarantee O(n log n))
def _merge(left, right, key_name, default, reverse):
    res = []
    i = j = 0
    while i < len(left) and j < len(right):
        lv = left[i].get(key_name, default)
        rv = right[j].get(key_name, default)
        if (lv <= rv and not reverse) or (lv > rv and reverse):
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    res.extend(left[i:]); res.extend(right[j:])
    return res

def sort_dicts_manual(arr, key_name, default=None, reverse=False):
    """
    Merge-sort implementation for list of dicts by key_name.
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = sort_dicts_manual(arr[:mid], key_name, default, reverse)
    right = sort_dicts_manual(arr[mid:], key_name, default, reverse)
    return _merge(left, right, key_name, default, reverse)


# Example usage
if __name__ == "__main__":
    data = [{"id": 3}, {"id": 1}, {"id": 2}, {"noid": 0}]
    print("AI sort:", sort_dicts_ai(data, "id", default=999))
    print("Manual sort:", sort_dicts_manual(data, "id", default=999))