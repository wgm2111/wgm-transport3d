# ====================================================================
#  Python-file
#     author:          William G. K. Martin
#     filename:        group.py
# ====================================================================

"""
Module containing class definition for the group used to simulate 
transport in three dimensions.  The two groups suppored, 'group24'
and 'group48'.
"""

# import libraries 
#___________________________________________________________________________
import sys
import scipy as sp

# Create dictionaries of group info for building groups of actions
#___________________________________________________________________________

group_dict = {}
group_dict['group24'] = {'actions':['', 'a', 'b', 'c', 
                                    'ab', 'bc', 'ac', 'abc', 
                                    's', 'as', 'bs', 'cs', 
                                    'abs', 'bcs', 'acs', 'abcs', 
                                    'ss', 'ass', 'bss', 'css', 
                                    'abss', 'bcss', 'acss', 'abcss'],
                         'rule_dict':{'sss':'',  
                                      'aa':'',   'bb':'',   'cc':'',
                                      'sc':'as', 'sa':'bs', 'sb':'cs', 
                                      'ba':'ab', 'ca':'ac', 'cb':'bc'},
                         'inst_dict':{'s':'spin', 
                                      'r':'yzx-spin',
                                      'a':'xreflect', 
                                      'b':'yreflect',
                                      'c':'zreflect'}}
                         
group_dict['group48'] = {'actions':['',    'a',    'b',    'c',    
                                    'ab',  'bc',   'ac',   'abc', 
                                    's',   'as',   'bs',   'cs',   
                                    'abs', 'bcs',  'acs',  'abcs', 
                                    'ss',  'ass',  'bss',  'css',  
                                    'abss','bcss', 'acss', 'abcss',
                                    'd',   'ad',   'bd',   'cd',   
                                    'abd', 'bcd',  'acd',  'abcd', 
                                    'e',   'ae',   'be',   'ce',   
                                    'abe', 'bce',  'ace',  'abce', 
                                    'f',   'af',   'bf',   'cf',   
                                    'abf', 'bcf',  'acf',  'abcf'], 
                         'rule_dict':{'sss':'', 
                                      'dd':'',   'ee':'',   'ff':'',
                                      'aa':'',   'bb':'',   'cc':'',
                                      'sc':'as', 'sa':'bs', 'sb':'cs', 
                                      'da':'ad', 'db':'cd', 'dc':'bd', 
                                      'ea':'ce', 'eb':'be', 'ec':'ae', 
                                      'fa':'bf', 'fb':'af', 'fc':'cf',
                                      'sd':'es', 'se':'fs', 'sf':'ds',
                                      'es':'f',  'ds':'e',  'fs':'d', 
                                      'ed':'ss', 'fe':'ss', 'df':'ss', 
                                      'de':'s',  'ef':'s',  'fd':'s',
                                      'ba':'ab', 'ca':'ac', 'cb':'bc'}, 
                         'inst_dict':{'s':'zxy-spin', 
                                      'r':'yzx-spin',
                                      'd':'xy-swap',
                                      'e':'xz-swap',
                                      'f':'yz-swap',
                                      'a':'x-reflect', 
                                      'b':'y-reflect',
                                      'c':'z-reflect'}}

def replace(old, new, str):
    """
    >>> replace('old', 'new', 'old')
    'new'
    >>> replace('old', 'new', 'oldold')
    'newnew'
    """
    return new.join(str.split(old))


# define classes
#___________________________________________________________________________
class Group(type([])):
    """ 
    Two groups are suppored, the order 24 and 48 group of symetries
    on the unit cube in three dimensions.
    """
    def __init__(self, options):
        """ initialize the group the group by specifying its name """
        super(Group, self).__init__(options.group_name)
        self.name = options.group_name
        self.actions = group_dict[self.name]['actions']
        self.order = len(self.actions)
        self.rule_dict = group_dict[self.name]['rule_dict']
        self.inst_dict = group_dict[self.name]['inst_dict']
        self.inverse_dict = {}
        
        # define a dictionary for defining action instructions
        self.act_inst = {}
        for act in self.actions:
            inst = []
            act_list = list(replace('ss','r', act)) # 'r' -> 'ss' for computing
            if not act_list:
                inst.append('do nothing')
            while act_list:
                last = act_list.pop()
                inst.append(self.inst_dict[last])
            self.act_inst[act] = ', '.join(inst)
      #  mix.GroupMixIn.__init__(self, opt.group_name)
        self[:] = self.actions
        
    # group opperations
    #___________________
    def compose(self, one, two):
        """ compose two actions in the order given and simplify with
        the dictionary of rules """
        long = one + two
        count = 0
        # loop until standard form (alphabedical order)
        while long not in self.actions:
            for outorder, inorder in self.rule_dict.iteritems():
                long = replace(outorder, inorder, long)
            count += 1        
        return long

    def inverse(self, one):
        """ Look for inverses, and store them as they are calculated. """
        if one in self.inverse_dict.keys():
            return self.inverse_dict[one]
        else:
            for elem in self.actions:
                if self.compose(elem, one) == '': break
            return elem

    
    def get_order(self, elem):
        """ function responsible for calculating and returning the order
        of any element in the group """
        orderTally = 1
        mult = elem
        while mult:
            mult = self.compose(mult, elem)
            orderTally+=1    
        return orderTally
    
    # print group information 
    #___________________________
    def print_inst_table(self):
        """ Print a table of instructions for performing each action in the group """
        print '='*80
        print '%-80s' %(self.name)
        print '_'*80
        print '%-30s %-50s' %('Action', 'Instruction')
        print '='*80
        for index, elem in enumerate(self.actions):
            print '%-30s %-50s' % (elem, self.act_inst[elem])

    def print_order_table(self):
        """ Print an order table for the group"""
        print '='*60
        print '%-60s' %(self.name)
        print '_'*60
        print '%-30s %-30s' %('Action', 'Order')
        print '='*60
        for index, elem in enumerate(self.actions):
            print '%-30s %-30d' % (elem, self.get_order(elem))

    def print_mult_table(self):
        print '='*9*self.order
        print '%-21s' %(self.name)
        print '='*9*self.order
        for i, row in enumerate(self.actions):
            print "|", '%-6s' %(row), 
            for j, col in enumerate(self.actions):
                if j == self.order-1:
                    print "|", '%-5s' %(self.compose(row, col)),
                elif j >=1:
                    print "|", '%-6s' %(self.compose(row, col)),
            print "|"
            if i == self.order-1:
                print '='*9*self.order
            else:
                print '_'*9*self.order


# Testing
#___________________________________________________________________________
if __name__ == '__main__':

    # import
    import sys
    sys.path.append('/Users/will/myCode/transport3d_py/src/classes')
    import options

    # condition on testing group
    test_group = True
    
    # Is the resulting set of actions a group?
    if test_group:
        o = options.Options('o', 10, 10, 'group24')
        g = Group(o)
        rinverse_bool = []
        linverse_bool = []
        for act in g.actions:
            rinverse_bool.append(g.compose(act, g.inverse(act))=='')
            linverse_bool.append(g.compose(g.inverse(act), act)=='')
        print "Right inverses for group24? \n\t", all(rinverse_bool)
        print "Left inverses for group24? \n\t", all(linverse_bool)

        o = options.Options('o', 10, 10, 'group48')
        g = Group(o)
        rinverse_bool = []
        linverse_bool = []
        for act in g.actions:
            rinverse_bool.append(g.compose(act, g.inverse(act))=='')
            linverse_bool.append(g.compose(g.inverse(act), act)=='')
        print "Right inverses for group48? \n\t", all(rinverse_bool)
        print "Left inverses for group48? \n\t", all(linverse_bool)
        
        print "==============================================================="
        print g.name + ': test output '
        print "---------------------------------------------------------------"
        print g.name + ' members:'
        print '\t', '%-15s \t %-15s' %('Member', 'Type')
        keylist = g.__dict__.keys()
        keylist.sort()
        for att in keylist:
            if att not in dir(list):
                print '\t', '%-15s' %(att), type(g.__dict__[att])
        print "---------------------------------------------------------------"
        g.print_order_table()
        g.print_inst_table()

