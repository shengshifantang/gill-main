
datasets = 'coco-cn_train coco-cn_val coco-cn_test'.split()

imset = [None] * len(datasets)
full = set()

for i,dataset in enumerate(datasets):
    imsetfile = '%s.txt' % dataset
    imset[i] = set(map(str.strip, open(imsetfile).readlines()))
    print 'number of images in %s: %d' % (dataset, len(imset[i]))
    full = full.union(imset[i])

print 'number of images in total: %d' % len(full)

for i in range(len(datasets)-1):
    for j in range(i+1, len(datasets)):
        common = imset[i].intersection(imset[j])
        print 'number of images overlapped between %s and %s: %d' % (datasets[i], datasets[j], len(common))

manual_translated_file = 'imageid.manually-translated-caption.txt'
subset = [x.split('#')[0] for x in open(manual_translated_file).readlines()]
subset = set(subset)

print len(imset[-1].intersection(subset)), len(imset[-1].difference(subset))


tag_file = 'imageid.human-written-tags.txt'
subset = [x.split()[0] for x in open(tag_file).readlines()]
subset = set(subset)
print len(subset), len(subset.difference(imset[0]).difference(imset[1]).difference(imset[2]))


print 'verifying sentence files'
lines1 = open('imageid.human-written-caption.bosonseg.txt').readlines()
lines2 = open('imageid.human-written-caption.txt').readlines()

imset_from_sent_file = set()
for x,y in zip(lines1,lines2):
    assert(x.split()[0] == y.split()[0])
    img_id = x.split()[0].split('#')[0]
    assert(img_id in full)
    imset_from_sent_file.add(img_id)

assert(len(imset_from_sent_file) == len(full))

print 'okay'
