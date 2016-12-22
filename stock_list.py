# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 21:40:31 2016

@author: Shruti
"""

class StockList:
      def createList(self):
              
        '''
        Capital Goods Companies (Top 10 market capital) (3)
        
           Tesla Motors, Inc.
           PACCAR Inc.
           Illumina, Inc.    
        '''
        capGoodsStock = {'Capital Goods': ['TSLA', 'PCAR', 'ILMN'], 'color': '#00FFFF'}
        
        '''
        Consumer Non-Durables (3)
            
            The Kraft Heinz Company
            Mondelez International, Inc.
            Monster Beverage Corporation
        
        Consumer Services (22)
        
            Amazon.com, Inc.
            Comcast Corporation
            Starbucks Corporation
            Charter Communications, Inc.
            Costco Wholesale Corporation
            Twenty-First Century Fox, Inc.
            Netflix, Inc.
            Twenty-First Century Fox, Inc.
            Marriott International
            Liberty Global plc
            Ross Stores, Inc.
            DISH Network Corporation
            Reilly Automotive, Inc.
            JD.com, Inc.
            Equinix, Inc.
            Sirius XM Holdings Inc.
            Paychex, Inc.
            Dollar Tree, Inc.
            Expedia, Inc.
            Viacom Inc.
            Ulta Salon, Cosmetics & Fragrance, Inc.
            Fastenal Company    
        '''
        consServNDStock = { 'Consumer Services':
                            ['KHC', 'MDLZ', 'MNST', 'AMZN', 'CMCSA', 'SBUX', 'CHTR',
                           'COST', 'FOX', 'NFLX', 'FOXA', 'MAR', 'LBTYA', 'ROST',
                           'DISH', 'ORLY', 'JD', 'EQIX', 'SIRI', 'PAYX', 'DLTR',
                           'EXPE', 'VIA', 'ULTA', 'FAST'],
                           'color': '#0080FF'                       
                          }

        '''
        Finance (7)
                     
            CME Group Inc.
            TD Ameritrade Holding Corporation
            Fifth Third Bancorp
            Northern Trust Corporation
            T. Rowe Price Group, Inc.
            Interactive Brokers Group, Inc.
            Huntington Bancshares Incorporated
        '''
        finStock = { 'Finance':
                    ['CME', 'AMTD', 'FITB', 'NTRS', 'TROW', 'IBKR', 'HBAN'],
                    'color': '#7F00FF'}
        
        '''
        Health Care (15)
        
            Amgen Inc.
            Gilead Sciences, Inc.
            Celgene Corporation
            Walgreens Boots Alliance, Inc.
            Biogen Inc.
            Express Scripts Holding Company
            Regeneron Pharmaceuticals, Inc.
            Alexion Pharmaceuticals, Inc.
            Intuitive Surgical, Inc.
            Vertex Pharmaceuticals Incorporated
            Incyte Corporation
            Mylan N.V.
            Shire plc
            BioMarin Pharmaceutical Inc.
            DENTSPLY SIRONA Inc.
        '''
        hCareStock = { 'Health Care':
                        ['AMGN', 'GILD', 'CELG', 'WBA', 'BIIB', 'ESRX', 'REGN', 'ALXN',
                         'ISRG', 'VRTX', 'INCY', 'MYL', 'SHPG', 'BMRN', 'XRAY' ],
                         'color': '#00FF80'
                     }             
        '''
        Miscellaneous (5)
         
            The Priceline Group Inc. 
            PayPal Holdings, Inc.
            eBay Inc.
            Ctrip.com International, Ltd.
            NetEase, Inc.
        '''
        miscStock = {'Miscellaneous':
                        ['PCLN', 'PYPL', 'EBAY', 'CTRP', 'NTES'],
                     'color': '#FF8000'
                    }

        '''
        Public Utilities (1)
        
            T-Mobile US, Inc.
        '''
        pubUtilStock = {'Public Utilities': ['TMUS'], 'color':'#808080'}    

        '''
        Technology (30)
        
            Apple Inc.
            Google
            Microsoft Corporation
            Facebook, Inc.
            Intel Corporation
            Cisco Systems, Inc.
            QUALCOMM Incorporated
            Texas Instruments Incorporated
            Broadcom Limited
            Adobe Systems Incorporated
            NVIDIA Corporation
            Baidu, Inc.
            Automatic Data Processing, Inc.
            Yahoo! Inc.
            Applied Materials, Inc.
            Cognizant Technology Solutions Corporation
            NXP Semiconductors N.V.
            Intuit Inc.
            Activision Blizzard, Inc
            Electronic Arts Inc.
            Fiserv, Inc.
            Analog Devices, Inc.
            Micron Technology, Inc.
            Western Digital Corporation
            Lam Research Corporation
            Cerner Corporation
            Autodesk, Inc.
            Symantec Corporation
            Linear Technology Corporation
            IHS Markit Ltd.
        '''
        techStock = { 'Technology':
                        ['AAPL', 'GOOGL', 'MSFT', 'FB', 'INTC', 'CSCO', 'QCOM', 'TXN',
                     'AVGO', 'ADBE', 'NVDA', 'BIDU', 'ADP', 'YHOO', 'AMAT', 'CTSH',
                     'NXPI', 'INTU', 'ATVI', 'EA', 'FISV', 'ADI', 'MU', 'WDC', 'LRCX',
                     'CERN', 'ADSK', 'SYMC', 'LLTC', 'INFO'],
                     'color' : '#994C00'
                    }

        '''
        Transportation (2)
        
            CSX Corporation
            American Airlines Group, Inc.
        '''
        transStock = {'Transportation': ['CSX','AAL'], 'color': '#660000' }

        #Total Stock List
        stocks = [capGoodsStock, consServNDStock, finStock, hCareStock, miscStock, pubUtilStock, techStock, transStock]
        
        return stocks