def partition_modified_sequence(modseq):
    
    NotAA = lambda x: (x < 65) | (x > 90)

    sequence = []
    p = 0

    #if modseq == '.N[0.9840]AINIEELFQGISR.':
    #    print()
    while p < len(modseq):
        character = modseq[p]
        hx = ord(character)

        # Pull out mod, in the form of a floating point number
        if NotAA(hx):
            mod_lst = []

            # N-terminal modifications precede the amino acid letter
            nterm = True if p == 2 else False

            # All numerals and mathematical symbols are below 65
            while NotAA(hx):
                mod_lst.append(character)
                p += 1

                # This will happen if we have a C-term modification
                if p == len(modseq):
                    break
                else:
                    character = modseq[p]
                    hx = ord(character)
            mod = "".join(mod_lst)

            # Get rid of absent terminal modifications, represented as period
            if mod == '.':
                continue
            elif mod[-1] == '.':
                mod = mod[:-1]

            # Add the amino acid to the end of the number if N-term
            if nterm:# & (mod != "(+57.02)"):
                # Leave the 57.02 with C
                if "(+57.02)" in mod:
                    sequence[0] = sequence[0] + "(+57.02)"
                    mod = mod[8:]

                # The modification stands as its own token at the beginning
                if len(mod) > 0:
                    sequence.insert(0, mod)

            # Grab the previously stored sequence AA and add modification to it
            else:
                sequence[-1] += mod

            p -= 1

        else:
            sequence.append(character)

        #if "" in sequence:
        #    print(sequence)

        p += 1

    return sequence
