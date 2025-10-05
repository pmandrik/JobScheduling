### Introduction
Tape is a sequential I/O technology with relatively high seek times to get to a random position on tape. Tape has
the additional characteristic of not being "always on" like a disk - a tape cartridge must first be mounted into a tape
drive before it can be read, and also a tape drive has a start-up time before it can reach its maximum I/O rate.

1. "Tape in the cloud—Technology developments and roadmaps supporting 80 TB cartridge capacities" (2019) https://doi.org/10.1063/1.5130404
2. "Magnetic Tape Storage Technology" (2024) https://dl.acm.org/doi/10.1145/3708997
3. "Performance evaluation of tape library systems" (2022) sciencedirect.com/science/article/abs/pii/S0166531622000232?via%3Dihub

### Vendors
https://research.ibm.com/publications?search=eyJ0eXBlIjoic2VhcmNoIiwidmFsdWUiOiJtYWduZXRpYyB0YXBlIn0

### Topics
#### Track Following
Reliable operation of tape storage devices requires high positioning accuracy of the servo control system under vibration conditions. 
The demand for increased storage density makes it increasingly more challenging to meet the track-following performance requirements 
especially in the presence of external vibration disturbances. In the recently introduced flangeless tape drives, 
the head positioning system has translational and rotational capabilities to be able to compensate for both the lateral tape 
motion and the head-to-tape skew arising from large tape excursions [1-2].
Timing-based servo patterns and their geometrical properties, i.e. azimuthal angle and subframe length affect
the position estimation resolution, the system delay and track-following performance [3].

1.  "Vibration compensation in tape drive track following using multiple accelerometers", 2013
   https://www.sciencedirect.com/science/article/pii/S1474667015362595
1. "Track-following in tape storage: Lateral tape motion and control", 2012
   https://www.sciencedirect.com/science/article/abs/pii/S0957415811001231?via%3Dihub
1. "Track-following system optimization for future magnetic tape data storage", 2021
   https://www.sciencedirect.com/science/article/abs/pii/S0957415821001331?via%3Dihub
1. Servo-Pattern Design and Track-Following Control for Nanometer Head Positioning on Flexible Tape Media (2012) http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6118309

#### Error codes
1. https://www.sciencedirect.com/science/article/abs/pii/S0166531625000355?via%3Dihub

#### Data placement & data prefeching
Data prefetching is a well-known technique for hiding read latency by requesting data before it is needed to move it from a high-latency medium (e.g., disk) 
to a low-latency one (e.g., main memory). State-of-the-art access prediction methods typically record access history of individual files, data objects, or data segments. 
However, in systems with large amounts of infrequently accessed (or cold) data, file-level access history is often unavailable for much of the data due to the low frequency of access.

1. "I/O Acceleration via Multi-Tiered Data Buffering and Prefetching" https://link.springer.com/article/10.1007/s11390-020-9781-1
2. "Data Prefetching for Large Tiered Storage Systems" https://ieeexplore.ieee.org/document/8215562

#### Speed matching
Backhitching is the condition that occurs when a data cartridge stops, reverses, and restarts motion. 
A backhitch is the result of a mismatch between the data rates of the connected server and the tape drive.
Modern tape drives can adjust their tape drive velocities between two or more read/write data rates to better match
the data rate demands of the host. This velocity changing may provide improvements in drive performance and total
backhitch counts. The transition from one tape velocity to another may involve a rate change backhitch which itself could impact performance. 
1. https://patentimages.storage.googleapis.com/57/1b/36/27495fe400778e/US7023651.pdf
2. https://www.redbooks.ibm.com/redbooks/pdfs/sg245946.pdf

### Software
1. HPSS https://hpss-collaboration.org/best-of-class-for-tape/
