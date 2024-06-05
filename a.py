def generate_restriction_map(dna_sequence, recognition_site):
    fragments = []
    recognition_sites = []
    current_position = 0
    
    while True:
        site_position = dna_sequence.find(recognition_site, current_position)
        if site_position == -1:
            break
        recognition_sites.append(site_position)
        fragment_size = site_position - current_position
        fragments.append(fragment_size)
        current_position = site_position + len(recognition_site)
    
    if current_position < len(dna_sequence):
        last_fragment_size = len(dna_sequence) - current_position
        fragments.append(last_fragment_size)
    
    return fragments, recognition_sites

# Example usage:
dna_sequence = "ATGATGAGCTAGCTAGCTGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"
recognition_site = "AGCT"
fragments, recognition_sites = generate_restriction_map(dna_sequence, recognition_site)

print("Restriction Map:")
print("Recognition sites:", recognition_sites)
print("Fragment sizes:", fragments)
