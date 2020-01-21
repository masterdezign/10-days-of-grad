module Shuffle ( shuffleIO )
  where

import Control.Monad
import Control.Monad.Random
import Data.Array.ST
import GHC.Arr

-- | Shuffles a list
-- https://wiki.haskell.org/Random_shuffle
-- >> evalRandIO $ shuffle [1..10]
-- [5,3,1,8,6,4,2,9,10,7]
shuffle :: RandomGen g => [a] -> Rand g [a]
shuffle xs = do
  let l = length xs
  rands <- take l `fmap` getRandomRs (0, l-1)
  let ar = runSTArray $ do
        ar' <- thawSTArray $ listArray (0, l-1) xs
        forM_ (zip [0..(l-1)] rands) $ \(i, j) -> do
            vi <- readSTArray ar' i
            vj <- readSTArray ar' j
            writeSTArray ar' j vi
            writeSTArray ar' i vj
        return ar'
  return (elems ar)

shuffleIO :: [a] -> IO [a]
shuffleIO = evalRandIO. shuffle
