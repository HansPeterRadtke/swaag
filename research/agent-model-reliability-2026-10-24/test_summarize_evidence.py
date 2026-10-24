import unittest
from summarize_evidence import summarize

class SummaryTests(unittest.TestCase):
    def one(self,encoding,score,gold=8,model='m1',seed=17):
        return {'trial':encoding+model,'kind':'relevance','model':model,'encoding':encoding,'input':{'id':'fixed-case','gold':gold},'seed':seed,'canonical_answer':score,'valid':True,'correct':score==gold,'total_elapsed_seconds':.1,'steps':[{'value':score}]}

    def test_matched_comparison_is_only_identical_case_model_seed(self):
        items=[self.one('number',3),self.one('category',8),self.one('number',8,model='m2')]
        a=summarize(items)
        self.assertEqual(len(a['relevance_matched_encoding_pairs']),1)
        self.assertEqual(len(a['cross_model_matched_input_pairs']),1)
        self.assertFalse(a['cross_model_matched_input_pairs'][0]['both_wrong'])

    def test_incomplete_is_not_a_model_error(self):
        x=self.one('category',8);x['incomplete']=True
        a=summarize([x,{'trial':'gate','reason':'busy'}])
        self.assertEqual(a['completed'],0)
        self.assertEqual(a['capacity_or_transport_stops'],2)

if __name__=='__main__':unittest.main()
