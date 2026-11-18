"""Curated legal contract clauses covering the 6 core contract types.
Provides a self-contained offline dataset for N-gram training and testing.
"""

CURATED_LEGAL_CLAUSES = [
    # Governing Laws
    "This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles.",
    "This Agreement and all claims or causes of action arising out of or relating to this Agreement shall be governed by the laws of the State of Delaware.",
    "This Agreement shall be construed in accordance with and governed by the domestic laws of the State of California without giving effect to any choice of law rule.",
    "This Agreement shall be interpreted and enforced in accordance with the substantive laws of the Commonwealth of Massachusetts.",
    "All questions concerning the construction, validity, enforcement and interpretation of this Agreement shall be governed by the internal laws of the State of Minnesota.",
    "This Agreement and any dispute arising from the performance or breach hereof shall be governed by and construed in accordance with the laws of the State of Texas.",
    "This Agreement shall be governed by and interpreted under the laws of the State of Illinois applicable to contracts made and to be performed entirely therein.",
    "This Agreement shall be deemed to be made under, and shall be construed in accordance with, the laws of the State of Florida.",
    "The validity, construction and performance of this Agreement shall be governed by the applicable laws of the State of Washington.",
    "This Agreement shall be construed and enforced in accordance with the laws of the State of New Jersey without reference to its conflicts of laws provisions.",

    # Notices
    "All notices, requests, demands and other communications hereunder shall be in writing and shall be deemed to have been duly given if delivered personally or mailed.",
    "Any notice or other communication required or permitted to be delivered to any party under this Agreement shall be in writing and shall be deemed properly given.",
    "All notices under this Agreement shall be in writing and sent by certified mail, return receipt requested, to the addresses set forth above.",
    "Notice shall be deemed effective when delivered in person, sent by confirmed facsimile transmission, or deposited with a recognized overnight courier service.",
    "Any notice required or permitted hereunder shall be given in writing and shall be deemed effectively given upon personal delivery or upon deposit with the postal service.",
    "All notices and other communications provided for herein shall be in writing and shall be delivered by hand or overnight courier service, or sent by facsimile.",
    "Notices shall be addressed to the parties at their respective principal offices as indicated on the signature page hereof.",
    "Any notice given pursuant to this Agreement shall be sent by registered mail, postage prepaid, or by nationally recognized overnight courier service.",
    "All notices required under this Agreement will be in writing, will reference this Agreement, and will be deemed given when delivered personally or transmitted electronically.",
    "Notice given by email shall be deemed received upon confirmation of receipt or on the next business day following transmission.",

    # Counterparts
    "This Agreement may be executed in any number of counterparts, each of which shall be deemed an original, but all of which together shall constitute one and the same instrument.",
    "This Agreement may be executed in two or more counterparts, each of which shall be deemed an original, and all of which together shall be considered one document.",
    "This Agreement may be executed by facsimile or electronic transmission and in separate counterparts, each of which shall be deemed an original instrument.",
    "Counterpart execution shall be effective upon exchange of signature pages delivered by facsimile or electronic mail.",
    "This Amendment may be executed in one or more counterparts, each of which shall be deemed to be an original copy and all of which together shall constitute one agreement.",
    "Facsimile and electronic signatures shall be deemed original signatures for all purposes of this Agreement.",
    "This Agreement may be executed in several counterparts, each of which will be an original and all of which will constitute one and the same agreement.",
    "Signatures of the parties transmitted by electronic mail in PDF format shall be treated as original signatures for all purposes.",
    "This Agreement may be executed by the parties hereto in separate counterparts, each of which when so executed and delivered shall be an original.",
    "Counterparts may be delivered via facsimile, electronic mail (including pdf) or other transmission method and any counterpart so delivered shall be deemed to have been duly and validly delivered.",

    # Entire Agreements
    "This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior negotiations, representations or agreements.",
    "This Agreement, together with the Exhibits attached hereto, contains the complete understanding and agreement of the parties regarding its subject matter.",
    "This Agreement embodies the entire agreement and understanding between the parties hereto and supersedes all prior agreements and understandings relating to such subject matter.",
    "This Agreement contains the final, complete and exclusive expression of the agreement between the parties with respect to the transaction contemplated herein.",
    "This Agreement represents the entire understanding between the parties and supersedes all prior agreements, whether written or oral, relating to the subject matter hereof.",
    "There are no oral or unwritten agreements between the parties regarding the subject matter of this Agreement, and this Agreement supersedes all prior discussions.",
    "This Agreement supersedes all prior negotiations, commitments, agreements and understandings between the parties regarding the transactions contemplated hereby.",
    "No representation, promise or inducement not included in this Agreement shall be binding upon either party hereto.",
    "This Agreement and the Shareholder Agreement represent the entire agreement of the parties with respect to the subject matter hereof and supersede all prior understandings.",
    "This Agreement sets forth the entire agreement of the parties and may not be amended except in writing signed by authorized representatives of both parties.",

    # Severability
    "If any provision of this Agreement is held to be invalid, illegal or unenforceable, the validity, legality and enforceability of the remaining provisions shall not be affected.",
    "In the event that any provision of this Agreement shall be deemed invalid or unenforceable, such invalidity shall not affect the validity of any other provision hereof.",
    "Should any clause, sentence, paragraph, subsection or Section of this Agreement be declared invalid or unenforceable, the remainder of this Agreement shall remain in full force.",
    "Whenever possible, each provision of this Agreement shall be interpreted in such manner as to be effective and valid under applicable law.",
    "If any court of competent jurisdiction finds any term of this Agreement to be invalid or unenforceable, such term shall be modified to the minimum extent necessary.",
    "Any provision of this Agreement that is prohibited or unenforceable in any jurisdiction shall, as to such jurisdiction, be ineffective to the extent of such prohibition without invalidating the remaining provisions.",
    "If any provision of this Agreement is found by a proper authority to be unenforceable or invalid, such unenforceability shall not render this Agreement unenforceable as a whole.",
    "The invalidity or unenforceability of any particular provision of this Agreement shall not affect the other provisions hereof, and the Agreement shall be construed in all respects as if such invalid provision were omitted.",
    "If any provision hereof is determined to be invalid or contrary to law, the remaining provisions of this Agreement shall remain enforceable and binding.",
    "To the extent permitted by applicable law, each party waives any provision of law that renders any provision hereof prohibited or unenforceable in any respect.",

    # Survival
    "The covenants, representations, warranties and indemnification obligations contained in this Agreement shall survive the closing and termination of this Agreement.",
    "The provisions of Sections 5, 7, 8 and 10 hereof shall survive the termination or expiration of this Agreement for any reason.",
    "The rights and obligations of the parties under the provisions of this Agreement which by their nature should survive termination shall so survive.",
    "All representations, warranties and covenants made herein shall continue in full force and effect following the execution and delivery of this Agreement.",
    "The obligations of confidentiality, indemnification and non-disclosure shall survive for a period of five years following the termination of this Agreement.",
    "All covenants, agreements, representations and warranties made by the Borrower herein shall survive the making of the loan and the execution and delivery of the notes.",
    "Survival of obligations: Neither the expiration nor the termination of this Agreement shall relieve any party from any liability for breach committed prior thereto.",
    "The confidentiality and proprietary rights provisions of this Agreement shall survive the termination or cancellation of this Agreement indefinitely.",
    "Termination of this Agreement shall not affect any accrued rights or liabilities of either party existing at the date of termination.",
    "The provisions of this Article shall survive any termination, cancellation or completion of the services contemplated under this Agreement.",
]


def get_default_corpus_text() -> str:
    """Returns the concatenated legal corpus text."""
    return "\n\n".join(CURATED_LEGAL_CLAUSES)
